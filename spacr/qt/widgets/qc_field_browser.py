"""Browse segmentation-QC fields and reversibly quarantine bad arrays.

The Measure banner already has every field verdict in its scorecards.  This
dialog turns those records into a triage loop: one implicated field at a
time, every object-type verdict together, the merged intensities under
toggleable mask outlines, and left/right/Q keyboard operation.

Nothing in this dialog touches the filesystem on the GUI thread.  Loading,
render preparation AND the quarantine move itself all run on workers, and
the 400 ms button poll answers out of :mod:`spacr.qt.path_probe` rather
than stat-ing -- see `QCFieldBrowser._file_state` for the freeze that
bought.  No mask is opened merely because the Measure screen itself was
shown.
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from dataclasses import field as _dc_field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import functools

import numpy as np
from PySide6.QtCore import QEvent, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPalette, QPixmap
from PySide6.QtWidgets import (
    QBoxLayout,
    QCheckBox,
    QStyle,
    QStyleOptionViewItem,
    QStyledItemDelegate,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...errors import ConfigurationError as _ConfigurationError
from ...qc_quarantine import (
    QUARANTINE_DIRNAME,
    quarantine_dir_for,
    quarantine_field,
    resolve_field_path,
    restore_field,
)
from .. import path_probe
from ..hidpi import device_ratio, follow_device_ratio, scaled_for
from ..i18n import current_language, tr
from ..job_runner import JobRunner
from .toggle import Toggle
from .zoom_view import ZoomableImageView

LOG = logging.getLogger("spacr.qt.qc_field_browser")

__all__ = [
    "QCFieldBrowser",
    "QCFieldImage",
    "QCFieldTarget",
    "QCFieldVerdict",
    "finding_targets",
    "load_qc_field",
    "render_qc_field",
    "targets_from_digest",
]


MAX_DISPLAY_EDGE = 1600

_MASK_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "cell": (45, 220, 105),
    "nucleus": (210, 80, 255),
    "pathogen": (255, 145, 45),
    "organelle": (35, 205, 235),
}


@dataclass(frozen=True)
class QCFieldVerdict:
    """One object type's persisted verdict for the field on screen.

    :param object_type: the object type the verdict is for, e.g. ``"cell"``;
        may be empty.
    :param severity: ``"ok"``, ``"warn"`` or ``"fail"``, as the field's QC
        recorded it.
    :param flags: the flags QC raised for this object type in this field.
    :param note: QC's free-text note, or empty.
    """

    object_type: str
    severity: str
    flags: Tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class QCFieldTarget:
    """One unique plate/field target and everything QC said about it.

    :param field: the field name, with any ``.npy`` suffix removed.
    :param plate_root: the project root of the plate the field belongs to.
    :param merged_dir: the folder of merged field arrays,
        ``<plate_root>/merged``.
    :param verdicts: each object type's verdict for this field, sorted by
        object type.
    :param reasons: the flags or kinds of the findings that point at this
        field, as ``<object_type>:<flag>`` when the finding names an object
        type.
    :param finding_texts: headlines of the findings that point at this field.
    """

    field: str
    plate_root: str
    merged_dir: str
    verdicts: Tuple[QCFieldVerdict, ...] = ()
    reasons: Tuple[str, ...] = ()
    finding_texts: Tuple[str, ...] = ()

    @property
    def audit_flags(self) -> Tuple[str, ...]:
        """Object-qualified flags/reasons written to the quarantine ledger."""
        values: List[str] = []
        for verdict in self.verdicts:
            values.extend(
                f"{verdict.object_type}:{flag}" if verdict.object_type else flag
                for flag in verdict.flags
            )
        values.extend(self.reasons)
        return tuple(dict.fromkeys(value for value in values if value))


@dataclass
class QCFieldImage:
    """Worker-safe display payload for one merged field."""

    path: str = ""
    intensities: Optional[np.ndarray] = None
    channel_names: Tuple[str, ...] = ()
    masks: Dict[str, np.ndarray] = _dc_field(default_factory=dict)
    quarantined: bool = False
    warnings: Tuple[str, ...] = ()
    error: str = ""


def _card_root(card: Any, digest: Any) -> str:
    """Find the project root a scorecard came from.

    :param card: the scorecard; its mask path is two levels below the root.
    :param digest: the digest it belongs to, used when the card names no
        path of its own.
    :returns: an absolute root, or ``""`` when neither knows one.
    """
    path = str(getattr(card, "path", "") or "")
    if path:
        return os.path.dirname(os.path.dirname(os.path.abspath(path)))
    return os.path.abspath(str(getattr(digest, "root", "") or ""))


def _field_plate(field: str) -> str:
    """Return the plate a field name belongs to.

    :param field: the field name.
    :returns: the plate, parsed properly where the segmentation-QC parser
        can read the name, and otherwise everything before the first
        underscore -- a convention that holds for every name spaCR writes
        itself.
    """
    try:
        from ...seg_qc import parse_field_name

        return str(parse_field_name(field).plate)
    except Exception:
        return str(field).split("_", 1)[0]


def _group_digest(digest: Any) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Collect a digest's per-field verdicts, keyed by root and field.

    The ``.npy`` suffix is stripped from field names so a verdict recorded
    against the mask file and one recorded against the field agree on one
    key.

    :param digest: the QC digest to read.
    :returns: ``{(root, field): {...}}`` with each field's verdicts.
    """
    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for card in list(getattr(digest, "scorecards", ()) or ()):
        root = _card_root(card, digest)
        for qc in list(getattr(card, "field_qcs", ()) or ()):
            name = str(getattr(qc, "field", "") or "").strip()
            if not name:
                continue
            if name.lower().endswith(".npy"):
                name = name[:-4]
            key = (root, name)
            group = groups.setdefault(key, {
                "verdicts": [], "reasons": [], "finding_texts": []})
            group["verdicts"].append(QCFieldVerdict(
                object_type=str(getattr(qc, "object_type", "") or ""),
                severity=str(getattr(qc, "severity", "") or "ok"),
                flags=tuple(str(flag) for flag in
                            (getattr(qc, "flags", ()) or ())),
                note=str(getattr(qc, "note", "") or ""),
            ))
    return groups


def _keys_for_finding(
    finding: Any,
    groups: Dict[Tuple[str, str], Dict[str, Any]],
) -> List[Tuple[str, str]]:
    """Find the fields a finding is about.

    A FLAG finding carries exact field names and only those match. A
    POSITIONAL finding does not: every field matching its plate and object
    type is part of the pattern and belongs in the browser, including the
    individually-clean ones -- which is what makes the pattern visible.

    :param finding: the finding to resolve.
    :param groups: the digest's fields, as :func:`_group_digest` returns.
    :returns: the matching ``(root, field)`` keys.
    """
    exact = {str(name)[:-4] if str(name).lower().endswith(".npy")
             else str(name)
             for name in (getattr(finding, "fields", ()) or ())}
    plate = str(getattr(finding, "plate", "") or "")
    object_type = str(getattr(finding, "object_type", "") or "")
    keys: List[Tuple[str, str]] = []
    for key, group in groups.items():
        _root, name = key
        if exact and name not in exact:
            continue
        if plate and _field_plate(name) != plate:
            continue
        if object_type and not any(
                verdict.object_type == object_type
                for verdict in group["verdicts"]):
            continue
        if exact or str(getattr(finding, "kind", "")) != "clean":
            keys.append(key)
    return keys


def targets_from_digest(digest: Any) -> Tuple[QCFieldTarget, ...]:
    """Return every unique field implicated by a digest, plate-aware.

    A field flagged for cell and nucleus appears once with two verdict rows.
    Positional findings, which have no per-field flag, contribute all fields
    in their plate/object-type group so the browser does not quietly omit the
    very pattern the banner asked the user to inspect.

    :param digest: the QC digest (a :class:`spacr.seg_qc.QCDigest`); its
        ``scorecards`` and ``findings`` are read by attribute.
    """
    groups = _group_digest(digest)
    wanted = {
        key for key, group in groups.items()
        if any(v.flags or v.severity in {"warn", "fail"}
               for v in group["verdicts"])
    }
    for finding in list(getattr(digest, "findings", ()) or ()):
        keys = _keys_for_finding(finding, groups)
        reason = str(getattr(finding, "flag", "") or
                     getattr(finding, "kind", "") or "finding")
        object_type = str(getattr(finding, "object_type", "") or "")
        qualified = f"{object_type}:{reason}" if object_type else reason
        text = str(getattr(finding, "headline", "") or "")
        for key in keys:
            wanted.add(key)
            if qualified not in groups[key]["reasons"]:
                groups[key]["reasons"].append(qualified)
            if text and text not in groups[key]["finding_texts"]:
                groups[key]["finding_texts"].append(text)

    targets: List[QCFieldTarget] = []
    for root, name in sorted(wanted, key=lambda item: (item[0], item[1])):
        group = groups[(root, name)]
        targets.append(QCFieldTarget(
            field=name,
            plate_root=root,
            merged_dir=os.path.join(root, "merged"),
            verdicts=tuple(sorted(
                group["verdicts"], key=lambda value: value.object_type)),
            reasons=tuple(group["reasons"]),
            finding_texts=tuple(group["finding_texts"]),
        ))
    return tuple(targets)


def finding_targets(
    digest: Any,
    finding: Any,
    targets: Optional[Sequence[QCFieldTarget]] = None,
) -> Tuple[QCFieldTarget, ...]:
    """Return the browser targets belonging to one rendered finding.

    :param digest: the QC digest (a :class:`spacr.seg_qc.QCDigest`); its
        ``scorecards`` and ``findings`` are read by attribute.
    :param finding: one finding of that digest; a flag finding matches its
        exact fields, a positional one every field of its plate and object
        type.
    :param targets: the targets to choose from; ``None`` builds them with
        :func:`targets_from_digest`.
    """
    groups = _group_digest(digest)
    keys = set(_keys_for_finding(finding, groups))
    return tuple(
        target for target in (
            tuple(targets) if targets is not None else targets_from_digest(digest)
        )
        if (target.plate_root, target.field) in keys
    )


def _mask_path(folder: str, field: str) -> Path:
    """Return where a field's mask file lives.

    :param folder: the object-type mask folder.
    :param field: the field name, without its suffix.
    :returns: the ``.npy`` path.
    """
    return Path(folder) / f"{field}.npy"


def _display_stride(shape: Sequence[int]) -> int:
    """Choose a subsampling stride that brings an image under the display cap.

    :param shape: the image's ``(height, width)``.
    :returns: the stride, never below 1 -- an image already under the cap is
        shown whole.
    """
    edge = max(int(shape[0]), int(shape[1]))
    return max(1, int(math.ceil(edge / float(MAX_DISPLAY_EDGE))))


def _load_mask(path: Path, stride: int, shape: Tuple[int, int]) -> np.ndarray:
    """Load a mask, subsampled, and check it against the image it labels.

    Memory-mapped and read without pickle: a mask file is data, and
    unpickling one would execute whatever it contained.

    :param path: the mask file.
    :param stride: subsampling stride.
    :param shape: the image's shape, which the mask must match.
    :returns: a 2-D label array, copied out of the map.
    :raises ValueError: if the mask is not 2-D after squeezing a singleton
        axis, or if its shape does not match the image -- outlines drawn
        from a mismatched mask would land on the wrong objects.
    """
    mask = np.load(str(path), mmap_mode="r", allow_pickle=False)
    if mask.ndim == 3 and 1 in mask.shape:
        mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {mask.shape}")
    if tuple(int(v) for v in mask.shape) != shape:
        raise ValueError(f"mask shape {mask.shape} does not match image {shape}")
    return np.array(mask[::stride, ::stride], copy=True)


def load_qc_field(
    target: QCFieldTarget,
    language: str = "en",
) -> QCFieldImage:
    """Read one target into a bounded display payload; safe on a worker.

    Separate mask stacks are preferred because those are the artifacts the
    scorecards date.  A merged-plane fallback supports plates whose stack
    folders have since been archived while the self-describing merged array
    remains.

    :param target: the field to read; its merged array is looked up in
        ``merged_dir`` (active or quarantined).
    :param language: UI language code for the error and warning messages.
    """
    merged_dir = target.merged_dir
    path = resolve_field_path(merged_dir, target.field)
    if path is None:
        return QCFieldImage(error=tr(
            "This field is already gone: no active or quarantined merged "
            "array exists for {field}.", language=language,
            field=target.field))

    warnings: List[str] = []
    try:
        array = np.load(str(path), mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        return QCFieldImage(
            path=str(path),
            quarantined=path.parent.name == "merged_quarantined",
            error=tr("Could not read {field}: {error}", language=language,
                     field=target.field, error=str(exc)),
        )
    if array.ndim != 3 or array.shape[2] < 1:
        return QCFieldImage(
            path=str(path),
            quarantined=path.parent.name == "merged_quarantined",
            error=tr(
                "Expected a merged (height, width, channels) array for "
                "{field}; found shape {shape}.",
                language=language,
                field=target.field, shape=str(getattr(array, "shape", None))),
        )

    from ...crops import DEFAULT_MASK_DIMS, read_merged_plane_layout
    from ...seg_qc import find_mask_stacks

    layout = None
    try:
        layout = read_merged_plane_layout(merged_dir)
    except Exception as exc:
        warnings.append(tr("Plane-layout metadata could not be read: {error}",
                           language=language,
                           error=str(exc)))
    stacks = find_mask_stacks(target.plate_root)
    height, width, planes = (int(v) for v in array.shape)
    stride = _display_stride((height, width))

    if layout is not None:
        raw_names = list(layout.get("intensity_channels") or ())
        intensity_count = min(len(raw_names), planes)
        channel_names = tuple(str(name) for name in raw_names[:intensity_count])
        mask_dims = dict(layout.get("mask_dims") or {})
    else:
        if stacks and planes > len(stacks):
            intensity_count = planes - len(stacks)
        elif planes > min(DEFAULT_MASK_DIMS.values()):
            intensity_count = min(DEFAULT_MASK_DIMS.values())
        else:
            intensity_count = planes
        intensity_count = max(1, min(intensity_count, planes))
        channel_names = tuple(str(index + 1)
                              for index in range(intensity_count))
        mask_dims = {
            name: dim for name, dim in DEFAULT_MASK_DIMS.items()
            if int(dim) < planes
        }

    intensities = np.array(
        array[::stride, ::stride, :intensity_count], copy=True)
    source_shape = (height, width)
    masks: Dict[str, np.ndarray] = {}
    object_types = set(stacks) | set(mask_dims)
    object_types.update(v.object_type for v in target.verdicts
                        if v.object_type)
    for object_type in sorted(object_types):
        stack_path = _mask_path(stacks.get(object_type, ""), target.field)
        try:
            if stacks.get(object_type) and stack_path.is_file():
                masks[object_type] = _load_mask(
                    stack_path, stride, source_shape)
                continue
            dim = mask_dims.get(object_type)
            if dim is not None and 0 <= int(dim) < planes:
                masks[object_type] = np.array(
                    array[::stride, ::stride, int(dim)], copy=True)
                continue
            warnings.append(tr("No {object_type} mask exists for this field.",
                               language=language,
                               object_type=object_type))
        except Exception as exc:
            warnings.append(tr(
                "Could not read the {object_type} mask: {error}",
                language=language,
                object_type=object_type, error=str(exc)))
    del array
    return QCFieldImage(
        path=str(path),
        intensities=intensities,
        channel_names=channel_names,
        masks=masks,
        quarantined=path.parent.name == "merged_quarantined",
        warnings=tuple(warnings),
    )


def _load_and_record_state(
    target: QCFieldTarget,
    language: str,
    paths: Tuple[str, str],
) -> QCFieldImage:
    """Load one field AND settle its two file states, both on the worker.

    The browser cannot ask the filesystem where a field is: that question ran
    every 400 ms on the GUI thread and, on a sleeping ``autofs`` mount, one
    stat took over twenty seconds -- see `QCFieldBrowser._file_state`. This
    job is already off the GUI thread and is already going to stat the active
    copy (`resolve_field_path` looks there first), so it answers both
    questions here and records the answers in `path_probe`.

    Doing it in the loading job rather than in its callback is what keeps the
    button EXACT rather than merely optimistic: by the time `_on_loaded`
    paints, the cache holds the truth about both copies, so a field that is
    in both folders still says so on the first frame, and one whose name is
    not a stem at all is still reported gone. The `is_symlink` exclusion the
    old inline check made survives here too; `path_probe` has no lstat
    variant of its own.

    A raising loader is returned as an error payload rather than allowed to
    escape, and that is not tidiness. `JobRunner.job_failed` carries no job
    id and is NOT generation-guarded, so a load abandoned by `cancel()` --
    the Measure banner re-pointing the dialog with `open_at` mid-load is the
    real path -- used to paint its own failure over the NEXT field's
    "Loading…" line. Coming back as a result instead puts the message behind
    the generation check that already drops stale results.
    """
    for path in paths:
        if not path:
            continue
        try:
            candidate = Path(path)
            present = candidate.is_file() and not candidate.is_symlink()
        except (OSError, ValueError):
            present = False
        path_probe.prime(path, present)
    try:
        return load_qc_field(target, language)
    except Exception as exc:                                     # noqa: BLE001
        LOG.info("could not load a QC field", exc_info=True)
        return QCFieldImage(error=tr("Could not load this field: {error}",
                                     language=language, error=str(exc)))


def _move_field(
    target: QCFieldTarget,
    was_quarantined: bool,
    sink: Dict[str, Any],
) -> Tuple[bool, str]:
    """Quarantine or restore one field. Safe on a worker; BLOCKS on disk.

    Every step of this is a filesystem call on a path the user chose:
    `quarantine_dir_for` alone is a ``Path.resolve()`` realpath walk over
    every component, and the move that follows adds ``mkdir``, ``stat``,
    ``link``, an ``fsync``-ed ledger write and an ``unlink``. On a sleeping
    ``/nas_mnt`` autofs share a single one of those had
    not returned after twenty seconds. It ran on the GUI thread until
    2026-09-04, which made the `path_probe` gate in `_sync_action` worth
    nothing: the poll no longer froze, and then pressing the button it guards
    froze the application anyway.

    :param sink: receives the committed outcome before the worker returns it
        to the matching GUI completion callback.
    :returns: ``(now_quarantined, destination)``.
    """
    if was_quarantined:
        destination = restore_field(
            quarantine_dir_for(target.merged_dir), target.field)
        outcome = (False, str(destination))
    else:
        destination = quarantine_field(
            target.merged_dir, target.field, flags=target.audit_flags)
        outcome = (True, str(destination))
    sink["outcome"] = outcome
    return outcome


def _render_or_message(
    payload: QCFieldImage,
    channel: int,
    layers: Tuple[str, ...],
    language: str,
) -> Tuple[Optional[np.ndarray], str]:
    """Render one field, returning a failure instead of raising it.

    Same reason as `_load_and_record_state`: a render abandoned by
    `_render_jobs.cancel()` must not be able to paint "Could not render this
    field" over the field that replaced it, and `job_failed` is the one
    completion path `JobRunner` does not generation-guard.
    """
    try:
        return render_qc_field(payload, channel, layers), ""
    except Exception as exc:                                     # noqa: BLE001
        LOG.info("could not render a QC field", exc_info=True)
        return None, tr("Could not render this field: {error}",
                        language=language, error=str(exc))


def _normalise(plane: np.ndarray) -> np.ndarray:
    """Scale one plane to 8-bit over its 2nd-98th percentile.

    The robust range rather than the full one: a single hot pixel otherwise
    takes the whole scale and the image goes black. A plane with no finite
    values, or no spread, comes back as zeros rather than as a division by
    zero.

    :param plane: the plane to scale.
    :returns: a uint8 array of the same shape.
    """
    values = np.asarray(plane)
    finite = values[np.isfinite(values)] if np.issubdtype(
        values.dtype, np.floating) else values.reshape(-1)
    if not finite.size:
        return np.zeros(values.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, (2.0, 98.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = (values.astype(np.float32) - float(lo)) * (255.0 / (hi - lo))
    return np.nan_to_num(scaled, nan=0.0, posinf=255.0, neginf=0.0).clip(
        0, 255).astype(np.uint8)


def _base_rgb(intensities: np.ndarray, channel: int) -> np.ndarray:
    """Build the RGB base a field's outlines are drawn over.

    One chosen channel is drawn grey; with no choice, a single-channel image
    is grey and a multi-channel one maps its first three channels to R, G
    and B.

    :param intensities: the field's channels, as ``(H, W, C)``.
    :param channel: which channel to show, or ``-1`` for the composite.
    :returns: a uint8 ``(H, W, 3)`` image.
    """
    if channel >= 0:
        mono = _normalise(intensities[..., channel])
        return np.repeat(mono[..., None], 3, axis=2)
    count = int(intensities.shape[2])
    if count == 1:
        mono = _normalise(intensities[..., 0])
        return np.repeat(mono[..., None], 3, axis=2)
    rgb = np.zeros(intensities.shape[:2] + (3,), dtype=np.uint8)
    for index in range(min(3, count)):
        rgb[..., index] = _normalise(intensities[..., index])
    return rgb


def _boundary(mask: np.ndarray) -> np.ndarray:
    """Return the one-pixel outline of every labelled object.

    A pixel is an edge where it is labelled and differs from a neighbour,
    plus the array's own border -- an object running off the edge of the
    field is outlined there too rather than appearing open.

    :param mask: the label array.
    :returns: a boolean array marking the outlines.
    """
    labels = np.asarray(mask)
    positive = labels > 0
    edge = np.zeros(labels.shape, dtype=bool)
    edge[0, :] |= positive[0, :]
    edge[-1, :] |= positive[-1, :]
    edge[:, 0] |= positive[:, 0]
    edge[:, -1] |= positive[:, -1]
    different = labels[1:, :] != labels[:-1, :]
    edge[1:, :] |= positive[1:, :] & different
    edge[:-1, :] |= positive[:-1, :] & different
    different = labels[:, 1:] != labels[:, :-1]
    edge[:, 1:] |= positive[:, 1:] & different
    edge[:, :-1] |= positive[:, :-1] & different
    return edge


def _mask_colour(object_type: str) -> Tuple[int, int, int]:
    """Pick a stable colour for an object type.

    Known types get their fixed colour and any organelle role shares one. A
    name spaCR does not know gets a colour derived from the name itself, so
    a custom role is vivid, distinct, and the same on every run.

    :param object_type: the role to colour.
    :returns: an RGB triple.
    """
    if object_type in _MASK_COLOURS:
        return _MASK_COLOURS[object_type]
    if object_type.startswith("organelle"):
        return _MASK_COLOURS["organelle"]
    seed = sum((index + 1) * ord(char)
               for index, char in enumerate(object_type))
    return (70 + seed % 170, 70 + (seed // 7) % 170,
            70 + (seed // 31) % 170)


def render_qc_field(
    payload: QCFieldImage,
    channel: int = -1,
    visible_masks: Iterable[str] = (),
) -> np.ndarray:
    """Render a uint8 RGB composite with object-type-coloured outlines.

    :param payload: a field loaded by :func:`load_qc_field`; one without
        intensities raises :class:`ValueError`.
    :param channel: the channel to show in grey, or -1 for the composite; a
        channel past the last raises :class:`IndexError`.
    :param visible_masks: the object types whose outlines are drawn; a type
        with no mask, or a mask of the wrong shape, is skipped.
    """
    if payload.intensities is None:
        raise ValueError(payload.error or "field has no image")
    count = int(payload.intensities.shape[2])
    channel = int(channel)
    if channel >= count:
        raise IndexError(f"channel {channel} is outside 0..{count - 1}")
    rgb = _base_rgb(payload.intensities, channel)
    for object_type in visible_masks:
        mask = payload.masks.get(object_type)
        if mask is None or mask.shape != rgb.shape[:2]:
            continue
        rgb[_boundary(mask)] = np.asarray(
            _mask_colour(object_type), dtype=np.uint8)
    return np.ascontiguousarray(rgb)


def _pixmap(rgb: np.ndarray) -> QPixmap:
    """Convert an RGB array to a ``QPixmap``.

    :param rgb: a uint8 ``(H, W, 3)`` image.
    :returns: the pixmap.
    """
    height, width = rgb.shape[:2]
    image = QImage(
        rgb.data, width, height, int(rgb.strides[0]), QImage.Format_RGB888)
    return QPixmap.fromImage(image.copy())


#: The QC browser's image canvas. It USED TO BE DEFINED HERE, as
#: ``_FieldView``; item 473 needed the same fit-zoom-pan view for its
#: raw-versus-enhanced window and moved it to
#: :mod:`spacr.qt.widgets.zoom_view` rather than write a second one. The
#: old name stays because this file reads well with it and because a name
#: in a test is a name worth not breaking; the class is the shared one.
_FieldView = ZoomableImageView


class QCFieldBrowser(QDialog):
    """Non-modal triage dialog for the flagged fields in one QC digest.

    :param targets: unique plate/field records from :func:`targets_from_digest`.
    :param initial_field: the field whose banner link was activated.
    :param initial_plate_root: disambiguates identical stems in two plates.
    :param run_active: callable returning whether Measure is in flight.  File
        mutation is disabled while it returns True.
    :param threaded: False is a test seam; production image loads are threaded.
    :param parent: parent widget; ownership only.
    """

    quarantineChanged = Signal(str, bool)

    #: The last answers `path_probe` actually gave for the field on screen,
    #: used as the DEFAULT for the next question about it. `_recheck_files`
    #: retires both keys every two seconds so a copy deleted from outside the
    #: application is still noticed, and between that retirement and the
    #: replacement probe landing -- up to `path_probe.PROBE_TIMEOUT_S` on the
    #: slow mount this whole exercise is about -- a fixed optimistic default
    #: would redraw a quarantined field as active, twice a minute, and let Q
    #: try to quarantine a file that is not there. Carrying the last answer
    #: forward makes the re-check invisible until it has something to say.
    #: Class attributes, so the browser answers correctly before ``__init__``
    #: has run -- the file-state probe is exercised on a bare instance.
    _last_active = True
    _last_quarantined = False

    def __init__(
        self,
        targets: Sequence[QCFieldTarget],
        *,
        initial_field: str = "",
        initial_plate_root: str = "",
        run_active: Optional[Callable[[], bool]] = None,
        threaded: bool = True,
        parent=None,
    ) -> None:
        """Build the browser over a list of fields to triage.

        :param targets: the fields to review.
        :param initial_field: which one to open first.
        :param initial_plate_root: the plate they belong to.
        :param run_active: whether a run is still writing to them.
        :param threaded: whether loading and rendering run on workers.
        :param parent: parent widget.
        """
        super().__init__(parent)
        self.setObjectName("QCFieldBrowser")
        self.setWindowTitle(tr("Segmentation QC field browser"))
        self.setModal(False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.resize(1040, 760)

        self._targets = tuple(targets)
        self._run_active = run_active or (lambda: False)
        self._index = 0
        self._payload: Optional[QCFieldImage] = None
        self._layer_checks: Dict[str, QCheckBox] = {}
        self._action_notice = ""
        self._jobs = JobRunner(
            self, threaded=threaded, app_key="segmentation QC field",
            user_visible=False)
        self._jobs.job_failed.connect(self._on_load_failed)
        self._jobs.busy_changed.connect(self._on_load_busy_changed)
        self._render_jobs = JobRunner(
            self, threaded=threaded, app_key="segmentation QC rendering",
            user_visible=False)
        self._render_jobs.job_failed.connect(self._on_render_failed)
        self._move_jobs = JobRunner(
            self, threaded=threaded, app_key="segmentation QC quarantine")
        self._move_jobs.job_failed.connect(self._on_move_failed)
        self._move_jobs.busy_changed.connect(self._on_move_busy_changed)
        self._pending_move: Optional[Dict[str, Any]] = None
        self._torn_down = False

        if initial_field:
            for index, target in enumerate(self._targets):
                if target.field == initial_field and (
                    not initial_plate_root
                    or os.path.abspath(target.plate_root)
                    == os.path.abspath(initial_plate_root)
                ):
                    self._index = index
                    break

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(9)

        navigation = QHBoxLayout()
        self._previous = QPushButton(tr("← Previous flagged field"), self)
        self._previous.setObjectName("GhostButton")
        self._previous.clicked.connect(self.previous_field)
        navigation.addWidget(self._previous)
        self._field_title = QLabel(self)
        self._field_title.setObjectName("PrerunTitle")
        self._field_title.setAlignment(Qt.AlignCenter)
        self._field_title.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Preferred)
        navigation.addWidget(self._field_title, 1)
        self._next = QPushButton(tr("Next flagged field →"), self)
        self._next.setObjectName("GhostButton")
        self._next.clicked.connect(self.next_field)
        navigation.addWidget(self._next)
        root.addLayout(navigation)

        self._verdict = QLabel(self)
        self._verdict.setObjectName("PrerunSub")
        self._verdict.setWordWrap(True)
        self._verdict.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self._verdict)

        controls = QHBoxLayout()
        image_label = QLabel(tr("Image:"), self)
        image_label.setObjectName("PrerunSub")
        controls.addWidget(image_label)
        self._channel = QComboBox(self)
        self._channel.setObjectName("QCFieldChannel")
        self._channel.setToolTip(tr(
            "Choose a single merged intensity channel or the first three "
            "channels as an RGB composite."))
        self._channel.currentIndexChanged.connect(self._render)
        controls.addWidget(self._channel)
        controls.addSpacing(12)
        self._layers_widget = QWidget(self)
        self._layers_widget.setObjectName("QCFieldLayers")
        self._layers_widget.setStyleSheet(
            "QWidget#QCFieldLayers { background: transparent; }")
        self._layers = QHBoxLayout(self._layers_widget)
        self._layers.setContentsMargins(0, 0, 0, 0)
        self._layers.setSpacing(8)
        controls.addWidget(self._layers_widget)
        controls.addStretch(1)
        root.addLayout(controls)

        self._view = _FieldView(self)
        self._view.setObjectName("QCFieldImage")
        self._view.setMinimumHeight(360)
        root.addWidget(self._view, 1)

        self._load_status = QLabel(self)
        self._load_status.setObjectName("PrerunNote")
        self._load_status.setWordWrap(True)
        self._load_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self._load_status)

        actions = QHBoxLayout()
        self._action_status = QLabel(self)
        self._action_status.setObjectName("PrerunAdvisory")
        self._action_status.setWordWrap(True)
        actions.addWidget(self._action_status, 1)
        self._quarantine = QPushButton(self)
        self._quarantine.setObjectName("DangerButton")
        self._quarantine.setToolTip(tr(
            "Move this merged .npy to merged_quarantined so later Measure "
            "runs skip it. Press Q to quarantine or restore."))
        self._quarantine.clicked.connect(self.toggle_quarantine)
        actions.addWidget(self._quarantine)
        close = QPushButton(tr("Close"), self)
        close.setObjectName("GhostButton")
        close.clicked.connect(self.close)
        actions.addWidget(close)
        root.addLayout(actions)

        from ..shortcuts import _bind_screen_key

        self._left_shortcut = _bind_screen_key(
            self, "Field browser", "Left", self.previous_field)
        self._right_shortcut = _bind_screen_key(
            self, "Field browser", "Right", self.next_field)
        self._q_shortcut = _bind_screen_key(
            self, "Field browser", "Q", self.toggle_quarantine)

        self._run_timer = QTimer(self)
        self._run_timer.setInterval(400)
        self._run_timer.timeout.connect(self._sync_action)
        self._run_timer.start()
        path_probe.probes.answered.connect(self._on_probe_answered)
        self.finished.connect(self._on_finished)
        self._recheck_timer = QTimer(self)
        self._recheck_timer.setInterval(2000)
        self._recheck_timer.timeout.connect(self._recheck_files)
        self._recheck_timer.start()

        for widget in self.findChildren(QWidget):
            widget.installEventFilter(self)

        if self._targets:
            self._show_target()
        else:
            self._field_title.setText(tr("No flagged fields"))
            self._verdict.setText(tr(
                "This QC digest does not identify a field to browse."))
            self._channel.setEnabled(False)
            self._previous.setEnabled(False)
            self._next.setEnabled(False)
            self._quarantine.setEnabled(False)

    @property
    def current_target(self) -> Optional[QCFieldTarget]:
        """The plate-aware field currently shown, or None."""
        if not self._targets:
            return None
        return self._targets[self._index]

    @property
    def current_field(self) -> str:
        """The current field stem, for tests and external status panels."""
        target = self.current_target
        return target.field if target is not None else ""

    def _show_target(self, *, preserve_notice: bool = False) -> None:
        """Show one field, loading it if it is not already in hand.

        :param preserve_notice: True to keep the current notice on screen,
            so a message about the LAST action is not wiped by simply moving.
        """
        target = self.current_target
        if target is None:
            return
        if not preserve_notice:
            self._action_notice = ""
        self._field_title.setText(tr(
            "{field}  ·  flagged field {index} of {total}",
            field=target.field, index=self._index + 1,
            total=len(self._targets)))
        self._previous.setEnabled(self._index > 0)
        self._next.setEnabled(self._index + 1 < len(self._targets))
        self._draw_verdict(target)
        self._payload = None
        self._view.clear_image()
        self._clear_layers()
        self._channel.clear()
        self._channel.setEnabled(False)
        self._load_status.setText(tr("Loading merged image and masks…"))
        self._last_active = True
        self._last_quarantined = False
        self._jobs.cancel()
        self._render_jobs.cancel()
        language = current_language()
        self._jobs.submit(
            lambda selected=target, code=language,
            paths=self._field_paths(): _load_and_record_state(
                selected, code, paths),
            self._on_loaded)
        self._sync_action()

    def _on_load_busy_changed(self, _busy: bool) -> None:
        """Enable or disable the controls while a load is running.

        :param _busy: the new state; re-read from the loader.
        """
        self._sync_navigation()
        self._sync_action()

    def _busy(self) -> bool:
        """True while a load or a quarantine move owns this field."""
        return self._jobs.is_busy() or self._move_jobs.is_busy()

    def _sync_navigation(self) -> None:
        """Enable next and previous only where there is somewhere to go."""
        busy = self._busy()
        self._previous.setEnabled(not busy and self._index > 0)
        self._next.setEnabled(
            not busy and self._index + 1 < len(self._targets))

    def _draw_verdict(self, target: QCFieldTarget) -> None:
        """Show the QC verdict recorded for one field.

        :param target: the field.
        """
        lines: List[str] = []
        for verdict in target.verdicts:
            flags = ", ".join(verdict.flags) if verdict.flags else tr("no flags")
            line = tr(
                "[{severity}] {object_type}: {flags}",
                severity=verdict.severity.upper(),
                object_type=verdict.object_type or tr("object"),
                flags=flags)
            if verdict.note:
                line += f" — {verdict.note}"
            lines.append(line)
        for finding in target.finding_texts:
            if finding and all(finding not in line for line in lines):
                lines.append(tr("Plate-level finding: {finding}",
                                finding=finding))
        self._verdict.setText("\n".join(lines) or tr(
            "This field is implicated by the plate-level QC finding."))

    def _clear_layers(self) -> None:
        """Drop the loaded planes and release their arrays.

        RELEASED EXPLICITLY because a field is several full-resolution planes,
        and holding the previous field's while the next loads doubles the peak
        for no benefit.
        """
        self._layer_checks.clear()
        while self._layers.count():
            item = self._layers.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _on_loaded(self, payload: QCFieldImage) -> None:
        """Render a field that has finished loading.

        :param payload: the loaded planes.
        """
        self._payload = payload
        if payload.error:
            self._load_status.setText(payload.error)
            self._view.clear_image()
            self._channel.setEnabled(False)
            self._sync_action()
            return
        self._channel.blockSignals(True)
        self._channel.clear()
        self._channel.addItem(tr("Composite (channels 1–3)"), -1)
        for index, name in enumerate(payload.channel_names):
            self._channel.addItem(tr(
                "Channel {index}: {name}", index=index + 1, name=name), index)
        self._channel.setCurrentIndex(0)
        self._channel.blockSignals(False)
        self._channel.setEnabled(True)
        self._clear_layers()
        for object_type in sorted(payload.masks):
            checkbox = QCheckBox(tr("{object_type} mask",
                                    object_type=object_type), self)
            colour = _mask_colour(object_type)
            checkbox.setStyleSheet(
                f"QCheckBox {{ color: rgb{colour}; background: transparent; }}")
            checkbox.setChecked(True)
            checkbox.setToolTip(tr(
                "Show or hide the {object_type} mask outline.",
                object_type=object_type))
            checkbox.toggled.connect(self._render)
            checkbox.installEventFilter(self)
            self._layers.addWidget(checkbox)
            self._layer_checks[object_type] = checkbox
        notes = list(payload.warnings)
        location = tr("Quarantined copy") if payload.quarantined else tr(
            "Active merged copy")
        notes.insert(0, tr("{location}: {path}",
                           location=location, path=payload.path))
        self._load_status.setText("\n".join(notes))
        self._render()
        self._sync_navigation()
        self._sync_action()

    def _on_load_failed(self, message: str) -> None:
        """Report a field that could not be read, and move on.

        :param message: what went wrong.
        """
        self._load_status.setText(tr("Could not load this field: {error}",
                                     error=message))
        self._sync_navigation()
        self._sync_action()

    def _render(self, *_args) -> None:
        """Composite the loaded planes into the displayed image."""
        payload = self._payload
        if payload is None or payload.intensities is None:
            return
        channel = self._channel.currentData()
        channel = -1 if channel is None else int(channel)
        visible = [name for name, checkbox in self._layer_checks.items()
                   if checkbox.isChecked()]
        self._render_jobs.cancel()
        self._render_jobs.submit(
            lambda data=payload, selected=channel, layers=tuple(visible):
            render_qc_field(data, selected, layers),
            self._on_rendered)

    def _on_rendered(self, rgb: np.ndarray) -> None:
        """Show a finished composite.

        :param rgb: the rendered image.
        """
        self._view.set_pixmap(_pixmap(rgb))

    def _on_render_failed(self, message: str) -> None:
        """Report a composite that could not be built.

        :param message: what went wrong.
        """
        self._load_status.setText(tr(
            "Could not render this field: {error}", error=message))

    def previous_field(self) -> None:
        """Move to the previous implicated field, if one exists."""
        if self._busy() or self._index <= 0:
            return
        self._index -= 1
        self._show_target()

    def next_field(self) -> None:
        """Move to the next implicated field, if one exists."""
        if self._busy() or self._index + 1 >= len(self._targets):
            return
        self._index += 1
        self._show_target()

    def open_at(self, field: str, plate_root: str = "") -> bool:
        """Reposition an existing browser at a banner-link target.

        :param field: the field name, without ``.npy``, as held in
            :attr:`QCFieldTarget.field`.
        :param plate_root: the plate's project root, compared as an absolute
            path; empty matches the first target with that field in any plate.
        """
        for index, target in enumerate(self._targets):
            if target.field == field and (
                not plate_root
                or os.path.abspath(target.plate_root)
                == os.path.abspath(plate_root)
            ):
                self._index = index
                self._show_target()
                return True
        return False

    def _is_run_active(self) -> bool:
        """Whether a run is still writing to these fields.

        TRIAGE MUST NOT MOVE A FILE A RUN IS WRITING, so the destructive
        actions are gated on this rather than on whether the file exists.

        :returns: True while a run is active.
        """
        try:
            return bool(self._run_active())
        except Exception:
            LOG.debug("could not read Measure run state", exc_info=True)
            return False

    @staticmethod
    def _paths_for(target: Optional[QCFieldTarget]) -> Tuple[str, str]:
        """The active and quarantined ``.npy`` paths, as text only.

        Built with string joins rather than :func:`quarantine_dir_for`,
        which resolves the plate folder: ``Path.resolve()`` is a realpath
        walk over every component, and on one ``/nas_mnt``
        autofs mount one component was enough to park the GUI thread.

        Takes its target as an argument rather than reading
        :attr:`current_target`, because the completion handlers of a move
        have to address the field the move was STARTED for; the user may
        have been sent elsewhere by a banner link since.
        """
        if target is None:
            return "", ""
        merged = os.path.normpath(target.merged_dir)
        name = f"{target.field}.npy"
        return (os.path.join(merged, name),
                os.path.join(os.path.dirname(merged),
                             QUARANTINE_DIRNAME, name))

    def _field_paths(self) -> Tuple[str, str]:
        """The two ``.npy`` paths of the field currently on screen."""
        return self._paths_for(self.current_target)

    def _file_state(self) -> Tuple[bool, bool]:
        """Whether the field is in ``merged``, in quarantine, or both.

        THIS RUNS EVERY 400 ms, from `_sync_action` on the GUI thread, for
        as long as the dialog is open. It used to be two `Path.is_file()`
        calls plus `is_quarantined`, and a stat on a sleeping autofs mount
        was measured at over twenty seconds on 2026-09-04 -- so a browser
        opened on a NAS plate froze the whole application a couple of
        times a second, with no traceback, because a stalled event loop is
        not a crash. `path_probe` answers from its cache and probes in the
        background; `_on_probe_answered` redraws when the answer lands.

        The starting defaults are chosen the way `file_list.py` chooses
        them: optimistic for the active copy, which is what this dialog
        already assumed of a field it was asked to show, and pessimistic for
        the quarantined one, so an unknown answer can never render the "both
        copies exist" dead end that disables the button outright.

        They are short-lived: `_load_and_record_state` settles the two keys
        exactly, on the loading worker, before the image it fetched is
        painted. Afterwards the default is whatever was last KNOWN about
        this field -- see :attr:`_last_active` for why a fixed default would
        make `_recheck_files` flicker the button twice a minute.
        """
        target = self.current_target
        if target is None:
            return False, False
        active_path, quarantined_path = self._paths_for(target)
        active = path_probe.exists(active_path, default=self._last_active)
        quarantined = path_probe.exists(
            quarantined_path, default=self._last_quarantined)
        self._last_active = active
        self._last_quarantined = quarantined
        return active, quarantined

    def _recheck_files(self) -> None:
        """Ask again, slowly, about the two copies nothing here moved.

        `_file_state` used to stat on every 400 ms tick, and that is how the
        button noticed a copy that vanished from outside the application --
        a crashed run tidying up, or the user deleting one of two duplicates
        so that the "resolve duplicates" dead end could clear itself. The
        probe cache has no expiry, so this is what puts that back: drop both
        keys and let the background probe answer them again.

        Nothing is dropped while an answer is still outstanding. A probe
        against a mount that has stopped responding parks a thread for up to
        `path_probe.PROBE_TIMEOUT_S`, and re-arming faster than they land
        would queue a new one every tick against a share that is not going
        to answer any of them.
        """
        paths = self._field_paths()
        if not paths[0] or any(path_probe.known(path) is None
                               for path in paths):
            return
        for path in paths:
            path_probe.forget(path)
        self._file_state()

    def _on_probe_answered(self, path: str, _present: bool) -> None:
        """Repaint the button when a background probe corrects the cache.

        The optimism in `_file_state` has a cost -- a field that is really
        gone is drawn as present until its probe lands -- and this is the
        half that pays it back.

        A BOUND METHOD, not a closure, and that is the whole point.
        `path_probe.probes` is process-wide and outlives every dialog, and
        this one is `WA_DeleteOnClose`: dismissing it with Escape goes
        through `QDialog.done`, which deletes the widget WITHOUT calling
        `closeEvent`, so no teardown hook of ours is guaranteed to run. Qt
        drops a connection to a bound method of a destroyed QObject by
        itself; a closure captured in an attribute would instead keep the
        Python wrapper alive around a dead C++ object and call into it.
        The guard below is for the emission already in flight when that
        happens.
        """
        try:
            if path in self._field_paths():
                self._sync_action()
        except RuntimeError:
            pass

    def _sync_action(self) -> None:
        """Enable the triage actions only when they are safe to take."""
        target = self.current_target
        if target is None:
            self._quarantine.setEnabled(False)
            return
        active, quarantined = self._file_state()
        if active and quarantined:
            self._quarantine.setText(tr("Resolve duplicate copies"))
            self._quarantine.setEnabled(False)
            self._action_status.setText(tr(
                "Both merged and merged_quarantined contain this field. "
                "Nothing will be overwritten."))
            return
        self._quarantine.setText(
            tr("Restore field (Q)") if quarantined
            else tr("Quarantine field (Q)"))
        if self._is_run_active():
            self._quarantine.setEnabled(False)
            self._action_status.setText(tr(
                "Measure is running. Stop or finish that run before changing "
                "which fields it can see."))
            return
        if self._move_jobs.is_busy():
            self._quarantine.setEnabled(False)
            return
        if self._jobs.is_busy():
            self._quarantine.setEnabled(False)
            self._action_status.setText(tr(
                "Wait for this field to finish loading before moving it."))
            return
        if not active and not quarantined:
            self._quarantine.setEnabled(False)
            self._action_status.setText(tr(
                "This field is already gone. The scorecard may be out of date."))
            return
        self._quarantine.setEnabled(True)
        if self._action_notice:
            self._action_status.setText(self._action_notice)
        else:
            self._action_status.setText(
                tr("Restoring puts this field back into later Measure runs.")
                if quarantined else tr(
                    "Quarantine is reversible. Masks stay where they are; only "
                    "the merged .npy moves."))

    def toggle_quarantine(self) -> None:
        """Quarantine or restore the current field; also bound to Q.

        The move runs on a worker. `quarantine_dir_for` is a `Path.resolve()`
        realpath walk, and the rename that follows adds a mkdir, a stat, a
        link, an ``fsync``-ed ledger write and an unlink -- on the plate path
        the user chose, which may be a sleeping ``autofs`` share. On the GUI
        thread these would freeze the window. See `_move_field`.
        """
        target = self.current_target
        if (target is None or self._is_run_active() or self._jobs.is_busy()
                or self._move_jobs.is_busy()):
            self._sync_action()
            return
        active, quarantined = self._file_state()
        if active == quarantined:
            self._sync_action()
            return
        sink: Dict[str, Any] = {
            "target": target, "was_quarantined": quarantined}
        self._pending_move = sink
        self._move_jobs.submit(
            lambda selected=target, was=quarantined, box=sink:
            _move_field(selected, was, box),
            lambda _outcome, box=sink: self._apply_move(box))
        if self._pending_move is sink:
            self._sync_action()

    def _apply_move(self, sink: Dict[str, Any]) -> None:
        """Record a completed move and reload the field from its new home.

        Runs on the GUI thread, behind `JobRunner`'s generation check.
        """
        if self._pending_move is sink:
            self._pending_move = None
        target = sink["target"]
        changed, destination = sink.pop("outcome")
        active_path, quarantined_path = self._paths_for(target)
        path_probe.prime(active_path, not changed)
        path_probe.prime(quarantined_path, changed)
        self.quarantineChanged.emit(target.field, changed)
        if target is not self.current_target:
            self._sync_action()
            return
        self._action_notice = tr(
            "Quarantined {field} at {path}. Later Measure runs will skip it.",
            field=target.field, path=destination) if changed else tr(
            "Restored {field} to {path}.",
            field=target.field, path=destination)
        self._show_target(preserve_notice=True)

    def _on_move_busy_changed(self, _busy: bool) -> None:
        """Enable or disable the controls while a move is running.

        :param _busy: the new state; re-read from the mover.
        """
        self._sync_navigation()
        self._sync_action()

    def _on_move_failed(self, message: str) -> None:
        """Report a rename that did not happen, and drop what it assumed.

        `job_failed` is the one completion path `JobRunner` does not
        generation-guard, so this checks the move it belongs to itself.
        """
        sink = self._pending_move
        self._pending_move = None
        target = sink["target"] if sink is not None else self.current_target
        for path in self._paths_for(target):
            path_probe.forget(path)
        if target is not self.current_target:
            self._sync_action()
            return
        self._action_notice = tr(
            "Could not change quarantine: {error}", error=message)
        self._sync_action()
        self._action_status.setText(self._action_notice)

    def _handle_triage_key(self, key: int) -> bool:
        """Apply a one-key triage verdict.

        THE KEYBOARD IS THE INTERFACE: triage is one decision per field over
        hundreds of fields, and a mouse round-trip per verdict is the
        difference between a session and an afternoon.

        :param key: the key pressed.
        :returns: True when the key was a verdict.
        """
        if key == Qt.Key_Left:
            self.previous_field()
            return True
        if key == Qt.Key_Right:
            self.next_field()
            return True
        if key == Qt.Key_Q:
            self.toggle_quarantine()
            return True
        return False

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt override
        """Watch the widgets this filter is installed on.

        :param watched: the object the event is for.
        :param event: the event.
        :returns: True to stop the event going further.
        """
        from ..shortcuts import _screen_event_key
        if (event.type() == QEvent.KeyPress
                and self._handle_triage_key(_screen_event_key(self, event))):
            event.accept()
            return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Keep the triage keys working when the dialog itself has focus.

        :param event: the key press; a triage key (Left and Right to move
            between fields, or a verdict key such as Q) is handled and
            accepted, anything else goes to the base class.
        """
        from ..shortcuts import _screen_event_key
        if self._handle_triage_key(_screen_event_key(self, event)):
            event.accept()
            return
        super().keyPressEvent(event)

    def _on_finished(self, _result: int) -> None:
        """Let go of everything process-wide, however the dialog was dismissed.

        `closeEvent` IS NOT A TEARDOWN HOOK HERE, which is the whole reason
        this exists. The dialog is `WA_DeleteOnClose`, and Escape goes
        through `QDialog.done`, which deletes the widget without ever raising
        a close event -- so `closeEvent` runs for a click on the frame and
        not for the key most people dismiss a dialog with. `finished` is
        emitted on both paths, while the object is still alive.

        What must not be left behind is the connection to
        `path_probe.probes`, which is process-wide and outlives every dialog.
        Qt drops a connection to a bound method of a destroyed QObject on its
        own, so this is belt and braces rather than the only defence -- but
        the timers are not, and a 400 ms poll left running against a
        half-destroyed dialog is a crash rather than a leak.

        Idempotent: `finished` and `closeEvent` both reach it on the click
        path, and disconnecting twice raises, which is why both are caught.
        """
        self._run_timer.stop()
        self._recheck_timer.stop()
        try:
            path_probe.probes.answered.disconnect(self._on_probe_answered)
        except (RuntimeError, TypeError):
            pass

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Retire an in-flight image load before Qt destroys the dialog.

        :param event: the close event; passed on to the base class after the
            timers and image-load workers are stopped.
        """
        self._run_timer.stop()
        self._recheck_timer.stop()
        try:
            path_probe.probes.answered.disconnect(self._on_probe_answered)
        except (RuntimeError, TypeError):
            pass
        self._render_jobs.shutdown()
        self._jobs.shutdown()
        super().closeEvent(event)


class _OverviewValidationError(_ConfigurationError, ValueError):
    """An explicit overview input constraint the user can correct."""


@dataclass(frozen=True)
class _OverviewField:
    """One field address and its explicitly identified channel files."""

    plate: str
    well: str
    row: str
    column: int
    field: str
    files: tuple
    unresolved: bool = False
    field_id: str = ""
    time_id: str = ""
    z_id: str = ""
    axes: tuple = ()
    source: str = ""
    plate_id: str = ""

    @property
    def identity(self):
        """Keep the original paths in the identity, including unresolved fields."""
        return (self.plate, self.well, self.field_id or self.field,
                self.files, self.time_id, self.z_id)

    @property
    def caption(self):
        """Trace a view back to its plate, well, field and source files."""
        return " / ".join((self.plate, self.well, self.field))


def _overview_plate_key(field):
    """The stable key of a field's plate: its plate folder and its name within it.

    ``field.plate`` is the plate's name in this session, unique across every
    loaded source and free to be renamed for display; this key is what saved
    state is filed under, so A01 of one plate never meets A01 of another.

    :param field: an :class:`_OverviewField`.
    :returns: ``(source, plate_id)``; ``source`` is ``""`` for a field indexed
        without one.
    """
    return (str(field.source or ""), str(field.plate_id or field.plate))


def _overview_image_key(field):
    """The stable key of one image: its plate key, well, field and acquisition axes.

    :param field: an :class:`_OverviewField`.
    :returns: ``(source, plate_id, well, field, time_id, z_id)``.
    """
    return _overview_plate_key(field) + (str(field.well), str(field.field_id or field.field),
                                         str(field.time_id or ""), str(field.z_id or ""))


def _overview_key_text(key):
    """A plate or image key as the text the saved view state files it under."""
    import json

    return json.dumps([str(part) for part in key], ensure_ascii=False)


_OVERVIEW_VIEW_STATE_VERSION = 1

_OVERVIEW_PLATES_VIEWS = (("single", "Single"), ("side", "Side by side"), ("stacked", "Stacked"))
_OVERVIEW_PLATES_VIEW_TOOLTIP = ("Show one plate at a time, or the shown plates at once, side by side or "
                                 "stacked; each keeps its own name, selection and collapsed rows and columns.")


def _overview_view_state_path(source):
    """Where the Overview's view state for ``source`` is saved."""
    return Path(source) / "results" / "overview_view.json"


def _overview_empty_view_state():
    """A view state with nothing in it.

    ``plates`` maps :func:`_overview_key_text` of a plate key to
    ``{"name": display name, "collapsed_rows": [...], "collapsed_columns": [...]}``
    (zero-based table indices). ``images`` maps the text of an image key to
    that image's display override, ``{"channels": {channel: {...}},
    "saturation": 1.0}``, where each channel's entry may hold ``black`` and
    ``white`` (0-1 display units), ``gamma``, ``brightness``, ``contrast`` and
    ``visible`` (see :func:`_overview_override_plane`). ``sources`` are the
    paths the Overview was given, so reopening the first one brings back every
    plate, and ``options`` holds ``use_subfolders`` and ``excluded_plates``.
    ``view`` holds the Plates view, ``{"plates_view": "single" | "side" |
    "stacked", "shown_plates": [...]}``, the shown plates as plate-key texts
    and empty for every plate.
    """
    return {"version": _OVERVIEW_VIEW_STATE_VERSION, "sources": [], "plates": {}, "images": {}, "options": {},
            "view": {}}


def _overview_load_view_state(source):
    """Read the saved view state for ``source``; a missing or unreadable file reads as empty.

    Runs off the GUI thread. A file written by a newer version is not read,
    so it is never overwritten with less than it held.

    :param source: the folder the Overview was opened on.
    :returns: the state, always with every top-level entry.
    """
    import json

    state = _overview_empty_view_state()
    path = _overview_view_state_path(source)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return state
    if not isinstance(loaded, dict) or int(loaded.get("version") or 0) > _OVERVIEW_VIEW_STATE_VERSION:
        return state
    for name in ("plates", "images", "options", "view"):
        if isinstance(loaded.get(name), dict):
            state[name] = loaded[name]
    if isinstance(loaded.get("sources"), list):
        state["sources"] = [str(value) for value in loaded["sources"]]
    return state


def _overview_save_view_state(source, state):
    """Write the view state for ``source`` atomically; runs off the GUI thread.

    :param source: the folder the Overview was opened on.
    :param state: the state, as :func:`_overview_empty_view_state` describes.
    :returns: the written path.
    """
    import json
    from ... import tabular

    path = _overview_view_state_path(source)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state, version=_OVERVIEW_VIEW_STATE_VERSION)
    tabular._publish(str(path), lambda pending: Path(pending).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"))
    return path


_OVERVIEW_SCOPES = ("image", "well", "plate", "all")
_OVERVIEW_SCOPE_IMAGES = 64
_OVERVIEW_SCOPE_PIXELS = 4096


def _overview_channel_scopes(value):
    """Read the normalization scope setting into ``{channel: scope}``.

    :param value: one scope for every channel (``'plate'``), or per channel
        (``'1=plate, 2=image'``).
    :returns: ``{"*": scope}`` for one value, else ``{channel text: scope}``;
        unknown scopes read as ``'image'``.
    """
    text = str(value or "image").strip().lower()
    if "=" not in text:
        return {"*": text if text in _OVERVIEW_SCOPES else "image"}
    scopes = {}
    for part in text.split(","):
        channel, _, scope = part.partition("=")
        if channel.strip():
            scope = scope.strip()
            scopes[channel.strip()] = scope if scope in _OVERVIEW_SCOPES else "image"
    return scopes


def _overview_scope_text(scopes):
    """Write ``{channel: scope}`` back as the setting's text, one value when all agree."""
    values = set(scopes.values())
    if len(values) <= 1:
        return next(iter(values), "image")
    return ", ".join(f"{channel}={scope}" for channel, scope in scopes.items())


def _overview_scope_of(scopes, channel):
    """The scope one channel is normalized over."""
    return scopes.get(str(channel), scopes.get("*", "image"))


def _overview_scope_group(field, scope):
    """The images a field shares its bounds with: its well, its plate, or every plate."""
    if scope == "well":
        return _overview_plate_key(field) + (str(field.well),)
    if scope == "plate":
        return _overview_plate_key(field)
    return ()


def _overview_scope_statistics(groups, percentiles, axes, cancelled=None, channel_percentiles=None):
    """Display bounds per channel over each group's sampled pixels; runs off the GUI thread.

    At most :data:`_OVERVIEW_SCOPE_IMAGES` evenly spaced images per group
    are read at 256 pixels, and at most :data:`_OVERVIEW_SCOPE_PIXELS`
    evenly spaced finite pixels of each, so a wide scope costs about as
    much as a few thumbnails.

    :param groups: ``{cache key: (channel, fields)}``.
    :param percentiles: ``(low, high)`` percentiles.
    :param axes: ``time_index``, ``z_mode`` and ``z_index``.
    :param channel_percentiles: ``{channel: (low, high)}`` for the channels
        with their own percentiles; the others use ``percentiles``.
    :returns: ``{cache key: (low, high)}`` in raw intensity units.
    """
    result = {}
    for key, (channel, fields) in groups.items():
        if cancelled is not None and cancelled.is_set():
            break
        fields = tuple(fields)
        step = max(1, math.ceil(len(fields) / _OVERVIEW_SCOPE_IMAGES))
        samples = []
        for field in fields[::step]:
            for _channel, _path, values in _overview_cached_raw_planes(field, (channel,), edge=256, **axes):
                finite = np.asarray(values, dtype=np.float64).ravel()
                finite = finite[np.isfinite(finite)]
                if finite.size:
                    samples.append(finite[::max(1, finite.size // _OVERVIEW_SCOPE_PIXELS)])
        if samples:
            pooled = np.concatenate(samples)
            low, high = np.percentile(pooled, (channel_percentiles or {}).get(channel, percentiles))
            result[key] = (float(low), float(high))
    return result


def _overview_try_save_view_state(source, state):
    """:func:`_overview_save_view_state`, answering an unwritable folder with its error text."""
    try:
        return _overview_save_view_state(source, state)
    except OSError as error:
        return str(error)


def _overview_override_plane(plane, settings):
    """Apply one channel's per-image display override to a 0-1 display plane.

    The last of the three display stages: global normalization, then the
    scope's channel limits, then this. ``black`` and ``white`` map linearly
    to 0 and 1, then ``gamma``, ``contrast`` about mid-grey and
    ``brightness`` are applied and the result clipped; ``visible`` False
    blanks the channel.

    :param plane: float display plane in 0-1.
    :param settings: the channel's override, or ``None``.
    :returns: the adjusted plane.
    """
    if not settings:
        return plane
    if settings.get("visible") is False:
        return np.zeros_like(plane)
    black, white = float(settings.get("black", 0.)), float(settings.get("white", 1.))
    plane = np.clip((plane - black) / max(white - black, np.finfo(np.float32).eps), 0., 1.)
    gamma = float(settings.get("gamma", 1.))
    if gamma > 0 and gamma != 1.:
        plane = plane ** (1. / gamma)
    plane = (plane - .5) * float(settings.get("contrast", 1.)) + .5 + float(settings.get("brightness", 0.))
    return np.clip(plane, 0., 1.)


def _overview_saturated(rgb, saturation):
    """Scale an RGB composite's saturation about its per-pixel luminance."""
    if saturation is None or float(saturation) == 1.:
        return rgb
    values = rgb.astype(np.float32)
    grey = (values @ np.asarray((.299, .587, .114), dtype=np.float32))[..., None]
    return np.clip(grey + (values - grey) * float(saturation), 0, 255).astype(np.uint8)


_OVERVIEW_OVERRIDE_DEFAULTS = {"black": 0., "white": 1., "gamma": 1., "brightness": 0.,
                               "contrast": 1., "visible": True}

_OVERVIEW_GLOBAL_LAYERS = (
    ("global_normalization", "Global normalization",
     "Ticked, this image is drawn with the Normalization and Advanced normalization settings; "
     "unticked, it is stretched between its own minimum and maximum."),
    ("global_enhancement", "Global enhancement",
     "Ticked, this image is drawn with the Enhancement settings; unticked, it is shown without enhancement."),
)
"""The per-image switches for the global layers: override key, menu caption and help."""


def _overview_clean_override(override):
    """Keep only what differs from the defaults in one image's display override.

    :param override: ``{"channels": {channel: {...}}, "saturation": s,
        "global_normalization": bool, "global_enhancement": bool}``, or ``None``;
        the two global switches are kept only when off.
    :returns: the override with string channel keys and no default values, or
        ``None`` when nothing differs, so an image that was reset has no entry.
    """
    channels = {}
    for channel, settings in ((override or {}).get("channels") or {}).items():
        kept = {name: value for name, value in dict(settings or {}).items()
                if name in _OVERVIEW_OVERRIDE_DEFAULTS and value != _OVERVIEW_OVERRIDE_DEFAULTS[name]}
        if kept:
            channels[str(channel)] = kept
    cleaned = {"channels": channels} if channels else {}
    saturation = (override or {}).get("saturation")
    if saturation is not None and float(saturation) != 1.:
        cleaned["saturation"] = float(saturation)
    for name, _caption, _tooltip in _OVERVIEW_GLOBAL_LAYERS:
        if (override or {}).get(name) is False:
            cleaned[name] = False
    return cleaned or None


def _overview_channel_override(override, channel):
    """One channel's full display settings, defaults filled in.

    :param override: the image's override, or ``None``.
    :param channel: the channel.
    :returns: every key of :data:`_OVERVIEW_OVERRIDE_DEFAULTS`.
    """
    channels = (override or {}).get("channels") or {}
    return dict(_OVERVIEW_OVERRIDE_DEFAULTS, **dict(channels.get(str(channel), channels.get(channel)) or {}))


def _overview_override_targets(fields, field, scope):
    """The images an override is copied to: the image's well, its plate, or every loaded image.

    :param fields: every loaded field.
    :param field: the image the override comes from.
    :param scope: ``'well'``, ``'plate'`` or ``'all'``.
    :returns: the image keys, in field order, without repeats.
    """
    plate = _overview_plate_key(field)
    return tuple(dict.fromkeys(
        _overview_image_key(other) for other in fields
        if scope == "all" or (_overview_plate_key(other) == plate
                              and (scope == "plate" or other.well == field.well))))


def _overview_copied_overrides(overrides, source, targets, channel=None):
    """Copy one image's override, or one channel of it, onto other images.

    :param overrides: ``{image key: override}``; not changed.
    :param source: the image key copied from.
    :param targets: the image keys copied to.
    :param channel: copy only this channel's settings, or ``None`` for the
        whole override, saturation included.
    :returns: the new mapping.
    """
    import copy

    origin = overrides.get(source) or {}
    result = dict(overrides)
    for key in targets:
        if key == source:
            continue
        if channel is None:
            merged = copy.deepcopy(origin)
        else:
            merged = copy.deepcopy(result.get(key) or {})
            channels = dict(merged.get("channels") or {})
            channels.pop(channel, None)
            channels.pop(str(channel), None)
            settings = (origin.get("channels") or {}).get(str(channel))
            if settings:
                channels[str(channel)] = dict(settings)
            merged["channels"] = channels
        cleaned = _overview_clean_override(merged)
        if cleaned is None:
            result.pop(key, None)
        else:
            result[key] = cleaned
    return result


def _overview_gamma_midpoint(black, white, gamma):
    """The input level a gamma maps to mid-grey, between the black and white points.

    :param black: the black point, 0-1.
    :param white: the white point, 0-1.
    :param gamma: the gamma; 1 puts the midpoint halfway.
    :returns: the midpoint, 0-1.
    """
    return black + (white - black) * .5 ** float(gamma)


def _overview_gamma_from_midpoint(black, white, middle):
    """The gamma that maps ``middle`` to mid-grey; the inverse of :func:`_overview_gamma_midpoint`.

    :param black: the black point, 0-1.
    :param white: the white point, 0-1.
    :param middle: the input level that should show as mid-grey.
    :returns: the gamma, kept within 0.05 to 10.
    """
    span = max(white - black, 1e-6)
    fraction = min(max((middle - black) / span, 1e-3), 1. - 1e-3)
    return float(min(max(math.log(fraction) / math.log(.5), .05), 10.))


def _overview_display_histograms(field, channels, percentiles, options):
    """Histograms of each channel's display plane before its per-image override.

    Runs off the GUI thread. The planes are the views' own display planes --
    global normalization and scope limits applied -- so the black and white
    points sit on what the image shows.

    :param field: the image.
    :param channels: the shown channels.
    :param percentiles: the global percentile stretch, or ``None``.
    :param options: the views' display options without ``image_overrides``.
    :returns: ``{channel: 256 counts}``, or the reason they cannot be read.
    """
    try:
        views = _overview_pixels(field, channels, percentiles, edge=512, channel_views=True, **options)
    except Exception as error:
        return str(error)
    return {channel: np.bincount(plane[..., 0].ravel(), minlength=256) for channel, plane in views}


def _overview_natural_key(value):
    """Order physical field and axis identifiers by their numeric components."""
    import re
    return tuple((1, int(part)) if part.isdigit() else (0, part.casefold())
                 for part in re.split(r"(\d+)", str(value)))


def _overview_well_layout(layout):
    """Normalise an explicit ``(rows, columns, order)`` well-index layout."""
    if not layout:
        return None, None, None
    rows, columns, order = (tuple(layout) + (None, None, None))[:3]
    return (int(rows) if rows else None, int(columns) if columns else None,
            order if order in ("row", "column") else None)


def _overview_index(source, pattern="", convention="auto", cancelled=None, source_scope="recursive",
                    well_layout=None):
    """Index paths without decoding pixels; never invent an unresolved well.

    A numeric well (a Yokogawa CQ1 ``W0049`` index) is placed with the
    explicit ``well_layout`` when given, else the acquisition XMLs next to or
    above the images, else the CQ1 24-column default.
    """
    import re
    from ...schema import parse_well, well_id
    from ...regex_infer import _metadata_pattern, _CQ1_FILENAME, _yokogawa_acquisition, _yokogawa_well
    from ..folder_metadata import _classify, _FIELD_RX, _CHANNEL_RX

    root = Path(source).resolve()
    if not root.is_dir():
        raise _OverviewValidationError(tr("Choose an existing image folder."))
    if source_scope not in ("folder", "recursive"):
        raise _OverviewValidationError(tr("Choose this folder or an explicit recursive source scope."))
    try:
        custom = re.compile(pattern, re.I) if pattern else None
    except re.error as exc:
        raise _OverviewValidationError(str(exc)) from exc
    if custom is not None and "wellID" not in custom.groupindex:
        raise _OverviewValidationError(tr("The filename regex must include a wellID group."))
    layout = _overview_well_layout(well_layout)
    acquisitions = {}
    groups = {}
    for directory, folders, names in os.walk(root):
        folders[:] = (sorted(p for p in folders if not p.startswith(".")
                            and p.lower() not in ("results", "measurements"))
                      if source_scope == "recursive" else [])
        for name in sorted(names):
            if cancelled is not None and cancelled.is_set():
                return ()
            path = Path(directory) / name
            if path.suffix.lower() not in _OVERVIEW_IMAGE_SUFFIXES:
                continue
            values = {}
            if custom is not None:
                match = custom.match(name)
            else:
                match = re.match(_metadata_pattern(convention, path.suffix[1:]), name, re.I)
                if match is None:
                    match = re.match(_CQ1_FILENAME, name, re.I)
                if match is None:
                    match = re.match(r"(?P<plateID>.+)_(?P<wellID>[A-Za-z]+\d+)_"
                                     r"(?:F|field)?(?P<fieldID>\d+)(?:_C(?P<chanID>\d+))?(?:\.|$)", name, re.I)
            if match is not None:
                values = {key: value for key, value in match.groupdict().items() if value}
            filename_plate = values.get("plateID")
            if custom is None:
                parent_parts = path.relative_to(root).parts[:-1]
                well_parent = None
                for position in reversed(range(len(parent_parts))):
                    token = parent_parts[position]
                    axis = _classify(token)
                    if axis == "well":
                        values.setdefault("wellID", token)
                        if well_parent is None:
                            well_parent = position
                    elif axis == "plate":
                        values.setdefault("plateID", str(Path(*parent_parts[:position + 1])))
                    elif axis == "field":
                        values.setdefault("fieldID", _FIELD_RX.match(token).group(1))
                    elif axis == "channel":
                        values.setdefault("chanID", _CHANNEL_RX.match(token).group(1))
                if "plateID" not in values and well_parent is not None:
                    values["plateID"] = str(root.joinpath(*parent_parts[:well_parent]))
                channel = _CHANNEL_RX.match(path.stem)
                if channel:
                    values.setdefault("chanID", channel.group(1))
            plate = str(values.get("plateID") or root)
            if custom is not None and not filename_plate:
                scope = list(path.relative_to(root).parts[:-1])
                while scope and _classify(scope[-1]) in ("well", "field", "channel"):
                    scope.pop()
                if scope:
                    plate = str(root.joinpath(*scope))
            if filename_plate:
                scope = list(path.relative_to(root).parts[:-1])
                while scope and _classify(scope[-1]) in ("well", "field", "channel"):
                    scope.pop()
                if not scope or scope[-1] != filename_plate:
                    scope.append(filename_plate)
                plate = "/".join(scope)
            well = str(values.get("wellID") or "").upper()
            if well.isdigit():
                if directory not in acquisitions:
                    acquisitions[directory] = _yokogawa_acquisition(directory)
                position = _yokogawa_well(int(well), acquisitions[directory], *layout)
                well = well_id(*position) if position else ""
            try:
                row, column = parse_well(well, strict=True)
                well = well_id(row, column)
                row = row[1:]
                column = int(column[1:])
            except (ValueError, TypeError):
                row, column = "", 0
                well = ""
            field = str(values.get("fieldID") or "")
            if not field:
                relative = path.relative_to(root)
                stem = path.stem
                if match is not None and values.get("chanID"):
                    start, end = match.span("chanID")
                    stem = name[:start] + "{channel}" + name[end:]
                parents = list(relative.parts[:-1])
                if parents and values.get("chanID") and _classify(parents[-1]) == "channel":
                    parents[-1] = "{channel}"
                    if _CHANNEL_RX.match(path.stem):
                        stem = "{channel}"
                field = str(Path(*parents, stem))
            unresolved = not well
            if unresolved:
                key = (plate, "", str(path.relative_to(root)), values.get("timeID", ""), values.get("sliceID", ""))
            else:
                key = (plate, well, field, values.get("timeID", ""), values.get("sliceID", ""))
            channel = str(values.get("chanID") or "")
            if channel.isdigit():
                channel = str(int(channel))
            groups.setdefault(key, [row, column, unresolved, []])[3].append(
                (channel, str(path)))
    result = []
    for (plate, well, field, time, plane), (row, column, unresolved, files) in sorted(groups.items(), key=lambda item: tuple(_overview_natural_key(value) for value in item[0])):
        label = field + (f" / T{time} / Z{plane}" if time or plane else "")
        axes = tuple((axis, value) for axis, value in (("T", time), ("Z", plane)) if value)
        result.append(_OverviewField(plate, well, row, column, label,
                                     tuple(sorted(files, key=lambda item: (_overview_natural_key(item[0]), item[1]))), unresolved,
                                     field, str(time), str(plane), axes, str(root), plate))
    return tuple(result)


_OVERVIEW_IMAGE_SUFFIXES = (".tif", ".tiff", ".npy", ".png", ".jpg", ".jpeg")


def _overview_source_paths(source):
    """Split an Overview source into its normalized paths, in order, without repeats.

    :param source: one path, a list of paths, or paths separated by ``;``.
    :returns: the absolute paths.
    """
    items = list(source) if isinstance(source, (list, tuple)) else [source]
    paths = []
    for item in items:
        for part in str(item or "").split(";"):
            part = part.strip()
            if part:
                path = os.path.abspath(os.path.expanduser(part))
                if path not in paths:
                    paths.append(path)
    return paths


def _overview_direct_images(folder):
    """How many images sit directly in ``folder``; hidden files are not counted."""
    try:
        with os.scandir(folder) as entries:
            return sum(1 for entry in entries if not entry.name.startswith(".") and entry.is_file()
                       and os.path.splitext(entry.name)[1].lower() in _OVERVIEW_IMAGE_SUFFIXES)
    except OSError:
        return 0


def _overview_plate_subfolders(folder):
    """The immediate subfolders of ``folder`` that hold images directly.

    spaCR's own output folders (``spacr.validate._PLATE_OUTPUT_FOLDERS``) and
    hidden folders are skipped.

    :returns: ``[(path, image count), ...]`` in natural name order.
    """
    from ...validate import _PLATE_OUTPUT_FOLDERS

    try:
        with os.scandir(folder) as entries:
            names = sorted((entry.name for entry in entries if entry.is_dir() and not entry.name.startswith(".")
                            and entry.name.lower() not in _PLATE_OUTPUT_FOLDERS), key=_overview_natural_key)
    except OSError:
        return []
    found = [(os.path.join(folder, name), _overview_direct_images(os.path.join(folder, name))) for name in names]
    return [(path, count) for path, count in found if count]


def _overview_has_nested_images(folder):
    """Whether any image sits anywhere below ``folder``, outside spaCR's output folders."""
    from ...validate import _PLATE_OUTPUT_FOLDERS

    for directory, folders, names in os.walk(folder):
        folders[:] = [name for name in folders if not name.startswith(".")
                      and name.lower() not in _PLATE_OUTPUT_FOLDERS]
        if any(os.path.splitext(name)[1].lower() in _OVERVIEW_IMAGE_SUFFIXES for name in names):
            return True
    return False


def _overview_detect_plates(source, use_subfolders=(), excluded=(), names=None, source_scope="folder",
                            cancelled=None):
    """Find the plates in one or more source paths.

    A folder with images directly in it is a plate; otherwise each immediate
    subfolder with images directly in it is one. A folder with both is a
    plate and is flagged as mixed; listed in ``use_subfolders`` its
    subfolders are the plates instead. With the recursive source scope, a
    folder whose images sit deeper is one plate, as before.

    :param source: the source paths (see :func:`_overview_source_paths`).
    :param use_subfolders: mixed folders whose subfolders are the plates.
    :param excluded: plate folders the user removed.
    :param names: ``{plate folder: display name}`` the user chose.
    :param source_scope: ``"folder"`` or ``"recursive"``.
    :param cancelled: an event that stops the walk.
    :returns: ``{"plates": [{"path", "name", "images", "mixed", "source"}],
        "messages": [...]}``; names are unique, the folder name by default and
        its parent's name before it when two folders share a name.
    """
    use_subfolders = {os.path.abspath(path) for path in use_subfolders}
    excluded = {os.path.realpath(path) for path in excluded}
    plates, messages = [], []
    for path in _overview_source_paths(source):
        if cancelled is not None and cancelled.is_set():
            break
        if not os.path.isdir(path):
            messages.append(tr("{path} is not a folder.", path=path))
            continue
        direct = _overview_direct_images(path)
        subfolders = _overview_plate_subfolders(path)
        if direct and not (subfolders and path in use_subfolders):
            found = [(path, direct, bool(subfolders))]
        elif subfolders:
            found = [(folder, count, False) for folder, count in subfolders]
        elif source_scope == "recursive" and _overview_has_nested_images(path):
            found = [(path, 0, False)]
        else:
            messages.append(tr("No images were found in {path} or in its immediate subfolders.", path=path))
            continue
        for folder, count, mixed in found:
            if os.path.realpath(folder) not in excluded and all(plate["path"] != folder for plate in plates):
                plates.append({"path": folder, "images": count, "mixed": mixed, "source": path})
    from collections import Counter

    taken = Counter(os.path.basename(plate["path"]) for plate in plates)
    used = set()
    for plate in plates:
        folder = Path(plate["path"])
        name = folder.name if taken[folder.name] == 1 else folder.parent.name + "/" + folder.name
        candidate, number = name, 2
        while candidate in used:
            candidate, number = f"{name} ({number})", number + 1
        used.add(candidate)
        plate["default_name"] = candidate
        plate["name"] = str((names or {}).get(plate["path"]) or candidate)
    return {"plates": plates, "messages": messages}


def _overview_plate_pattern(patterns, folder, pattern):
    """The filename regex one plate folder is read with: its own, or ``pattern``.

    :param patterns: ``{plate folder: regex}``; folders are compared by real path.
    :param folder: the plate folder.
    :param pattern: the regex every other plate uses, the first plate's.
    """
    if not patterns or not folder:
        return pattern
    wanted = os.path.realpath(str(folder))
    return next((str(value) for key, value in patterns.items()
                 if value and os.path.realpath(str(key)) == wanted), pattern)


def _overview_index_sources(source, pattern="", convention="auto", cancelled=None, source_scope="recursive",
                            well_layout=None, use_subfolders=(), excluded=(), folder_names=False, patterns=None):
    """Index every plate of one or more source paths.

    One path that is itself the only plate is indexed as before, so its
    plate names come from the filenames, unless ``folder_names`` asks for
    the folder's name, as for plates added to an open Overview. Otherwise each plate folder is
    indexed on its own and its fields are named after the folder (and after
    the filename plate as well, when one folder holds several), so A01 of
    one plate never merges with A01 of another. Every field records its
    plate folder (``source``) and its plate name within it (``plate_id``).

    :param patterns: ``{plate folder: regex}`` for plates read with their
        own filename regex; every other plate uses ``pattern``.
    :returns: ``(fields, detection)``; ``detection`` is
        :func:`_overview_detect_plates`'s answer, whose messages say why no
        plate was found when ``fields`` is empty.
    :raises _OverviewValidationError: when no source path is a folder.
    """
    import dataclasses

    paths = _overview_source_paths(source)
    if not paths:
        raise _OverviewValidationError(tr("Choose an existing image folder."))
    if not any(os.path.isdir(path) for path in paths):
        raise _OverviewValidationError(tr("Choose an existing image folder."))
    detection = _overview_detect_plates(paths, use_subfolders, excluded, None, source_scope, cancelled)
    plates = detection["plates"]
    if not plates:
        return (), detection
    fields = []
    single = not folder_names and len(paths) == 1 and len(plates) == 1 and plates[0]["path"] == paths[0]
    for plate in plates:
        if cancelled is not None and cancelled.is_set():
            return (), detection
        scope = source_scope if (single or not plate["images"]) else "folder"
        own = _overview_plate_pattern(patterns, plate["path"], pattern)
        found = _overview_index(plate["path"], own, "auto" if own != pattern else convention, cancelled,
                                source_scope=scope, well_layout=well_layout)
        identities = sorted({field.plate for field in found}, key=_overview_natural_key)
        for field in found:
            if single:
                name = field.plate
            elif len(identities) == 1:
                name = plate["default_name"]
            else:
                name = plate["default_name"] + " / " + (Path(field.plate).name if os.path.isabs(field.plate)
                                                        else field.plate)
            fields.append(dataclasses.replace(field, plate=name))
    return tuple(fields), detection


def _overview_folders_dialog(parent, title):
    """A folder picker for one or several plate folders that opens at a sensible size and resizes by its edges.

    The non-native picker opened at Qt's small default, so its buttons were
    cut off, and four fifths of the screen was far too big. It opens at
    960 by 640, at most three fifths of a smaller screen, never smaller than
    its own layout needs so every button shows, and keeps a margin wider than
    the resize band, so every edge can be dragged the way every spaCR dialog
    is resized.

    :param parent: the widget the picker belongs to.
    :param title: the translated window title.
    :returns: the configured, not yet shown :class:`QFileDialog`.
    """
    from PySide6.QtCore import QSize
    from PySide6.QtWidgets import QAbstractItemView, QApplication, QFileDialog, QListView, QTreeView
    from .glass import RESIZE_BAND, let_the_user_resize

    dialog = QFileDialog(parent, title)
    dialog.setFileMode(QFileDialog.Directory)
    dialog.setOption(QFileDialog.DontUseNativeDialog, True)
    dialog.setOption(QFileDialog.ShowDirsOnly, True)
    for view in dialog.findChildren(QListView) + dialog.findChildren(QTreeView):
        view.setSelectionMode(QAbstractItemView.ExtendedSelection)
    margin = RESIZE_BAND + 4
    if dialog.layout() is not None:
        dialog.layout().setContentsMargins(margin, margin, margin, margin)
    screen = (parent.screen() if parent is not None else None) or QApplication.primaryScreen()
    wanted = QSize(960, 640)
    if screen is not None:
        available = screen.availableGeometry()
        wanted = wanted.boundedTo(QSize(int(available.width() * .6), int(available.height() * .6)))
        wanted = wanted.expandedTo(dialog.minimumSizeHint()).boundedTo(available.size())
    else:
        wanted = wanted.expandedTo(dialog.minimumSizeHint())
    dialog.resize(wanted)
    dialog.setMinimumSize(dialog.minimumSizeHint())
    dialog.setSizeGripEnabled(True)
    let_the_user_resize(dialog)
    return dialog


def _overview_load_limit(value):
    """Read a Max wells or Max fields setting: ``None`` for all, else a positive count."""
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip().lower()
    if text in ("", "all", "none", "0"):
        return None
    try:
        return max(1, int(float(text)))
    except ValueError:
        return None


def _overview_limited_fields(fields, max_wells=None, max_fields=None):
    """The fields loaded under the Max wells per plate and Max fields per well caps.

    Deterministic: each plate keeps its first ``max_wells`` wells in plate
    order (A01, A02, ..., B01, ...), each well its first ``max_fields``
    fields in field order, with every time point and Z plane of a kept
    field. Fields without a well are kept.

    :param fields: the indexed fields.
    :param max_wells: wells per plate, or ``None`` for all.
    :param max_fields: fields per well, or ``None`` for all.
    :returns: the kept fields, in their original order.
    """
    if max_wells is None and max_fields is None:
        return tuple(fields)
    wells, kept_fields = {}, {}
    for field in fields:
        if field.unresolved or not field.well:
            continue
        wells.setdefault(field.plate, set()).add(field.well)
        kept_fields.setdefault((field.plate, field.well), set()).add(field.field_id or field.field)
    allowed_wells = {plate: set(sorted(names, key=_overview_natural_key)[:max_wells] if max_wells else names)
                     for plate, names in wells.items()}
    allowed_fields = {address: set(sorted(names, key=_overview_natural_key)[:max_fields] if max_fields else names)
                      for address, names in kept_fields.items()}
    return tuple(field for field in fields if field.unresolved or not field.well or (
        field.well in allowed_wells[field.plate]
        and (field.field_id or field.field) in allowed_fields[(field.plate, field.well)]))


def _overview_tiff_pixel_size(image):
    """Read a micrometre pixel size from OME or ImageJ TIFF metadata, or None.

    A resolution in inches is ignored: it is a print density, usually the
    72 or 96 DPI default, not a pixel size.
    """
    import re
    try:
        description = image.pages[0].description or ""
    except (AttributeError, IndexError):
        description = ""
    if "PhysicalSizeX" in description:
        match = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', description)
        unit = re.search(r'PhysicalSizeXUnit="([^"]+)"', description)
        scale = {"µm": 1., "um": 1., "micron": 1., "nm": 1e-3, "mm": 1e3}.get(
            unit.group(1) if unit else "µm")
        if match and scale:
            value = float(match.group(1)) * scale
            return value if value > 0 else None
    try:
        tags = image.pages[0].tags
        resolution = tags["XResolution"].value
        unit = tags["ResolutionUnit"].value if "ResolutionUnit" in tags else 2
    except (KeyError, IndexError, AttributeError):
        return None
    numerator, denominator = resolution if isinstance(resolution, tuple) else (resolution, 1)
    if not numerator or not denominator:
        return None
    pixels_per_unit = numerator / denominator
    imagej = getattr(image, "imagej_metadata", None) or {}
    if str(imagej.get("unit", "")).casefold().replace("\\u00b5", "µ") in ("micron", "um", "µm"):
        return 1. / pixels_per_unit
    if int(unit) == 3:
        return 1e4 / pixels_per_unit
    return None


class _OverviewCache:
    """A thread-safe least-recently-used cache bounded by entry count and bytes.

    Thumbnails, previews and Original images are decoded on worker threads;
    a display change (channels, levels, normalization, enhancement) then
    re-renders from the decoded planes instead of reading the files again.
    """

    def __init__(self, entries, budget_bytes):
        """Create an empty cache.

        :param entries: the most entries kept.
        :param budget_bytes: the most array bytes kept.
        """
        import threading
        from collections import OrderedDict

        self._lock = threading.Lock()
        self._values = OrderedDict()
        self._entries, self._budget, self._bytes = entries, budget_bytes, 0

    @staticmethod
    def _size(value):
        """The array bytes a cached value holds."""
        if isinstance(value, np.ndarray):
            return value.nbytes
        if isinstance(value, tuple):
            return sum(item.nbytes for item in value if isinstance(item, np.ndarray))
        return 0

    def get(self, key):
        """Return the cached value, or ``None``, and mark it recently used."""
        with self._lock:
            value = self._values.get(key)
            if value is not None:
                self._values.move_to_end(key)
            return value

    def put(self, key, value):
        """Store ``value``, evicting the least recently used beyond the bounds."""
        size = self._size(value)
        if size > self._budget:
            return
        with self._lock:
            old = self._values.pop(key, None)
            if old is not None:
                self._bytes -= self._size(old)
            self._values[key] = value
            self._bytes += size
            while self._values and (len(self._values) > self._entries or self._bytes > self._budget):
                _key, evicted = self._values.popitem(last=False)
                self._bytes -= self._size(evicted)

    def clear(self):
        """Forget every cached value."""
        with self._lock:
            self._values.clear()
            self._bytes = 0


_OVERVIEW_INFO_CACHE = _OverviewCache(8192, 1 << 62)
_OVERVIEW_PLANE_CACHE = _OverviewCache(4096, 768 * 1024 ** 2)


def _overview_file_stamp(field):
    """The field's files with their size and modification time, for cache keys."""
    stamp = []
    for _channel, path in field.files:
        try:
            status = os.stat(path)
        except OSError:
            return None
        stamp.append((path, status.st_size, status.st_mtime_ns))
    return tuple(stamp)


def _overview_channel_info(field):
    """Inspect declared axes and channel IDs once per unchanged file set.

    :param field: the overview field.
    :returns: the channel and axis description of its files.
    """
    stamp = _overview_file_stamp(field)
    key = (field.identity, stamp)
    if stamp is not None:
        known = _OVERVIEW_INFO_CACHE.get(key)
        if known is not None:
            return known
    info = _overview_read_channel_info(field)
    if stamp is not None:
        _OVERVIEW_INFO_CACHE.put(key, info)
    return info


_OVERVIEW_DISK_BUDGET = 2 * 1024 ** 3
_OVERVIEW_DISK_PRUNE_EVERY = 256
_OVERVIEW_DISK_WRITES = [0]


def _overview_disk_cache_folder():
    """Where preview planes persist between sessions: the user's cache, never a source folder.

    ``SPACR_OVERVIEW_DISK_CACHE=0`` turns the disk cache off.

    :returns: the folder, or ``None`` when the disk cache is off.
    """
    if os.environ.get("SPACR_OVERVIEW_DISK_CACHE", "1").strip() in ("0", "false", "no", "off"):
        return None
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "spacr" / "overview_planes"


def _overview_disk_name(key):
    """The cache file for one preview plane key."""
    import hashlib

    folder = _overview_disk_cache_folder()
    if folder is None:
        return None
    digest = hashlib.sha1(repr(key).encode("utf-8", "replace")).hexdigest()
    return folder / digest[:2] / (digest + ".npy")


def _overview_disk_plane(key, path):
    """Read a persisted preview plane, or ``None``.

    The file name is a digest of the whole key, source file stamps included,
    so a changed source file is never served from the cache.

    :param key: the plane's cache key.
    :param path: the source file the plane belongs to.
    :returns: ``(path, read-only array)`` or ``None``.
    """
    name = _overview_disk_name(key)
    if name is None:
        return None
    try:
        array = np.load(name, allow_pickle=False)
    except (OSError, ValueError):
        return None
    array.setflags(write=False)
    return path, array


def _overview_store_disk_plane(key, path, array):
    """Persist one preview plane in the user's cache folder, atomically.

    :param key: the plane's cache key, including its source file stamp.
    :param path: the source file the plane came from.
    :param array: the downsampled raw plane.
    """
    name = _overview_disk_name(key)
    if name is None:
        return
    try:
        name.parent.mkdir(parents=True, exist_ok=True)
        pending = name.with_name(name.name + f".{os.getpid()}.{id(array)}.tmp")
        with open(pending, "wb") as handle:
            np.save(handle, np.ascontiguousarray(array), allow_pickle=False)
        os.replace(pending, name)
    except OSError:
        LOG.debug("could not persist an overview preview plane", exc_info=True)
        return
    _OVERVIEW_DISK_WRITES[0] += 1
    if _OVERVIEW_DISK_WRITES[0] % _OVERVIEW_DISK_PRUNE_EVERY == 0:
        _overview_prune_disk_cache(name.parent.parent)


def _overview_prune_disk_cache(folder):
    """Remove the oldest persisted planes beyond :data:`_OVERVIEW_DISK_BUDGET`."""
    try:
        entries = [(item.stat().st_mtime, item.stat().st_size, item) for item in folder.glob("*/*.npy")]
    except OSError:
        return
    total = sum(size for _mtime, size, _item in entries)
    for _mtime, size, item in sorted(entries):
        if total <= _OVERVIEW_DISK_BUDGET:
            break
        try:
            item.unlink()
            total -= size
        except OSError:
            continue


def _overview_cached_raw_planes(field, channels, *, edge=None, **axes):
    """Yield raw planes like :func:`_overview_raw_planes`, decoding each one once.

    Planes are kept per file stamp, channel, axis selection and edge, read-only,
    so display-only changes re-render without touching the files again.

    :param field: the overview field.
    :param channels: the channels to read, in order.
    :param edge: the bounded preview edge, or ``None`` for original pixels.
    :param axes: ``time_index``, ``z_mode`` and ``z_index``.
    """
    stamp = _overview_file_stamp(field)
    if stamp is None:
        yield from _overview_raw_planes(field, channels, edge=edge, **axes)
        return
    selection = tuple(sorted(axes.items()))
    keys = {channel: (field.identity, stamp, channel, selection, edge) for channel in channels}
    found = {channel: _OVERVIEW_PLANE_CACHE.get(key) for channel, key in keys.items()}
    files = {str(key): path for key, path in field.files}
    for channel in channels:
        if found[channel] is None and edge is not None:
            stored = _overview_disk_plane(keys[channel], files.get(str(channel), field.files[0][1]))
            if stored is not None:
                found[channel] = stored
                _OVERVIEW_PLANE_CACHE.put(keys[channel], stored)
    missing = tuple(channel for channel in channels if found[channel] is None)
    if missing:
        for channel, path, array in _overview_raw_planes(field, missing, edge=edge, **axes):
            array = np.array(array, copy=True)
            array.setflags(write=False)
            found[channel] = (path, array)
            _OVERVIEW_PLANE_CACHE.put(keys[channel], (path, array))
            if edge is not None:
                _overview_store_disk_plane(keys[channel], path, array)
    for channel in channels:
        path, array = found[channel]
        yield channel, path, array


def _overview_read_channel_info(field):
    """Inspect declared image axes and channel IDs without decoding pixels."""
    from PIL import Image

    records = []
    pixel_size = None
    separate = any(key for key, _ in field.files)
    if (not field.files or (not separate and len(field.files) != 1)
            or (separate and any(not key for key, _ in field.files))):
        raise _OverviewValidationError(tr("This field has ambiguous image files."))
    for key, path in field.files:
        suffix = Path(path).suffix.lower()
        if suffix == ".npy":
            image = np.load(path, mmap_mode="r", allow_pickle=False)
            shape, dtype = image.shape, image.dtype
            axes = {2: "YX", 3: "YXC", 4: "ZYXC", 5: "TZYXC"}.get(image.ndim, "")
        elif suffix in (".tif", ".tiff"):
            import tifffile
            try:
                with tifffile.TiffFile(path) as image:
                    series = image.series[0]
                    shape, dtype, axes = series.shape, series.dtype, series.axes
                    pixel_size = _overview_tiff_pixel_size(image)
            except getattr(tifffile, "TiffFileError", ValueError) as error:
                raise _OverviewValidationError(tr("This file is not a readable TIFF image: {name}",
                                                  name=Path(path).name)) from error
        else:
            with Image.open(path) as image:
                bands = len(image.getbands())
                shape = (image.height, image.width) + ((bands,) if bands > 1 else ())
                axes = "YXC" if bands > 1 else "YX"
                from PIL import ImageMode
                dtype = np.dtype(ImageMode.getmode(image.mode).typestr)
        axes = str(axes).upper().replace("S", "C")
        if (len(axes) != len(shape) or len(set(axes)) != len(axes)
                or set(axes) - set("TZCYX") or axes.replace("T", "").replace("Z", "").replace("C", "") != "YX"):
            raise _OverviewValidationError(tr("The image channel axis is unresolved."))
        channel_axis = axes.find("C")
        records.append(dict(path=path, file_channel=key, shape=tuple(shape),
                            axes=axes, dtype=str(np.dtype(dtype)),
                            channel_axis=channel_axis,
                            channel_count=shape[channel_axis] if channel_axis >= 0 else 1,
                            time_count=shape[axes.index("T")] if "T" in axes else 1,
                            z_count=shape[axes.index("Z")] if "Z" in axes else 1,
                            pixel_size_um=pixel_size))
        pixel_size = None
    if separate:
        ids = [int(record["file_channel"]) for record in records
               if str(record["file_channel"]).isdigit()]
        available = tuple(sorted(key for key in set(ids) if ids.count(key) == 1
                                 and next(record for record in records
                                          if str(record["file_channel"]).isdigit()
                                          and int(record["file_channel"]) == key)["channel_count"] == 1))
    else:
        available = tuple(range(records[0]["channel_count"]))
    return dict(kind="files" if separate else "stack", channels=available,
                files=tuple(records))


def _overview_plane_region(array, region):
    """Slice spatial source pixels before materialization and decode accounting."""
    if region is None:
        return array
    y0, y1, x0, x1 = (int(value) for value in region)
    if not 0 <= y0 < y1 <= array.shape[-2] or not 0 <= x0 < x1 <= array.shape[-1]:
        raise _OverviewValidationError(tr("The requested image region lies outside the field."))
    return array[..., y0:y1, x0:x1]


_OVERVIEW_SPARSE_STRIDE = 4


_OVERVIEW_READ_TRIALS = 4
_OVERVIEW_READ_TIMES = {}
_OVERVIEW_READ_LOCK = threading.Lock()


def _overview_device(path):
    """The storage device a file lives on, for choosing how to read it."""
    try:
        return os.stat(path).st_dev
    except OSError:
        return None


def _overview_read_method(path):
    """Choose sparse row reads or whole-page reads for a file's storage.

    Reading only a thumbnail's rows moves far fewer bytes, which wins on a
    network share or a spinning disk; on fast local storage one sequential
    read of the whole page can be quicker than many small ones. The first
    reads on each device try both and the faster one is kept.

    :param path: the TIFF file.
    :returns: ``"sparse"``, ``"full"`` or, during the trial, either.
    """
    device = _overview_device(path)
    with _OVERVIEW_READ_LOCK:
        times = _OVERVIEW_READ_TIMES.setdefault(device, {"sparse": [], "full": []})
        sparse, full = times["sparse"], times["full"]
        if len(sparse) < _OVERVIEW_READ_TRIALS or len(full) < _OVERVIEW_READ_TRIALS:
            return "sparse" if len(sparse) <= len(full) else "full"
        return "sparse" if sum(sparse) / len(sparse) <= sum(full) / len(full) else "full"


def _overview_record_read(path, method, seconds):
    """Remember how long one thumbnail read took on a file's device."""
    device = _overview_device(path)
    with _OVERVIEW_READ_LOCK:
        times = _OVERVIEW_READ_TIMES.setdefault(device, {"sparse": [], "full": []})[method]
        times.append(seconds)
        del times[:-16]


def _overview_sparse_stride(page, edge):
    """The row stride for reading a thumbnail straight from an uncompressed page.

    :param page: the TIFF page.
    :param edge: the thumbnail's longest edge in pixels.
    :returns: the stride, or 0 when the whole page should be read instead.
    """
    if not page.is_memmappable or len(page.shape) != 2 or not page.is_contiguous:
        return 0
    stride = max(1, math.ceil(max(page.shape) / max(1, edge)))
    return stride if stride >= _OVERVIEW_SPARSE_STRIDE else 0


def _overview_sparse_rows(path, page, stride, byteorder):
    """Read only every ``stride``-th row and column of an uncompressed page.

    A thumbnail needs a small fraction of the rows; reading just those with
    positioned reads, read-ahead off, moves that fraction of the bytes
    instead of the whole image -- what matters on a network share.

    :param path: the TIFF file.
    :param page: its uncompressed, contiguous two-dimensional page.
    :param stride: the row and column step.
    :param byteorder: the file's byte order, ``<`` or ``>``.
    :returns: the strided plane as a new array.
    """
    contiguous = page.is_contiguous
    offset = contiguous[0] if isinstance(contiguous, tuple) else page.dataoffsets[0]
    height, width = page.shape
    dtype = np.dtype(page.dtype).newbyteorder(byteorder)
    row_bytes = width * dtype.itemsize
    rows = range(0, height, stride)
    plane = np.empty((len(rows), width), dtype=dtype)
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    positioned = getattr(os, "pread", None)
    try:
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_RANDOM)
        for index, row in enumerate(rows):
            if positioned is not None:
                data = positioned(descriptor, row_bytes, offset + row * row_bytes)
            else:
                os.lseek(descriptor, offset + row * row_bytes, os.SEEK_SET)
                data = os.read(descriptor, row_bytes)
            if len(data) != row_bytes:
                raise _OverviewValidationError(tr("This TIFF layout cannot be read as bounded image planes."))
            plane[index] = np.frombuffer(data, dtype=dtype)
    finally:
        os.close(descriptor)
    return np.ascontiguousarray(plane[:, ::stride]).astype(dtype.newbyteorder("="), copy=False)


def _overview_tiff_plane(record, channel, selections, z_mode, edge=None, region=None, page_cache=None):
    """Read validated TIFF axes as selected pages, streaming explicit Z projections."""
    import tifffile

    axes = record["axes"]
    page_cache = {} if page_cache is None else page_cache
    with tifffile.TiffFile(record["path"]) as tif:
        series = tif.series[0]
        page_axes = series.pages[0].axes.upper().replace("S", "C")
        if not axes.endswith(page_axes):
            raise _OverviewValidationError(tr("This TIFF layout cannot be read as bounded image planes."))
        leading = axes[:-len(page_axes)]
        leading_shape = series.shape[:len(leading)]
        z_values = (range(record["z_count"]) if "Z" in axes and z_mode != "plane"
                    else (selections.get("Z", 0),))
        result, count = None, 0
        for z in z_values:
            positions = dict(selections, C=channel, Z=z)
            coordinates = tuple(positions[axis] for axis in leading)
            page_index = int(np.ravel_multi_index(coordinates, leading_shape)) if leading else 0
            page = series.pages[page_index]
            sparse = (edge is not None and region is None and page_axes == "YX"
                      and _overview_sparse_stride(page, edge))
            method = _overview_read_method(record["path"]) if sparse else None
            if method == "full":
                sparse = 0
            page_key = record["path"], page.index, sparse
            if page_key not in page_cache:
                page_cache.clear()
                started = time.perf_counter()
                if sparse:
                    plane = _overview_sparse_rows(record["path"], page, sparse, tif.byteorder)
                elif method is not None:
                    plane = np.array(tifffile.memmap(record["path"], page=page.index, mode="r")[
                        ::_overview_sparse_stride(page, edge), ::_overview_sparse_stride(page, edge)])
                elif page.is_memmappable:
                    plane = tifffile.memmap(record["path"], page=page.index, mode="r")
                else:
                    if math.prod(page.shape) * np.dtype(page.dtype).itemsize > 128 * 1024 ** 2:
                        raise _OverviewValidationError(tr("This image exceeds the 128 MiB preview decode limit."))
                    plane = page.asarray()
                if method is not None:
                    _overview_record_read(record["path"], method, time.perf_counter() - started)
                page_cache[page_key] = plane
            plane = page_cache[page_key]
            selector = tuple(positions[axis] if axis in "TCZ" else slice(None)
                             for axis in page_axes)
            plane = _overview_plane_region(plane[selector], region)
            if edge is not None and not sparse and method is None:
                stride = max(1, math.ceil(max(plane.shape) / max(1, edge)))
                plane = plane[::stride, ::stride]
            if plane.nbytes > 128 * 1024 ** 2:
                raise _OverviewValidationError(tr("This image exceeds the 128 MiB preview decode limit."))
            if z_mode == "plane" or "Z" not in axes:
                return plane
            if result is None:
                result_dtype = np.dtype(np.float64 if z_mode == "mean" else plane.dtype)
                if plane.size * result_dtype.itemsize > 128 * 1024 ** 2:
                    raise _OverviewValidationError(tr("This image exceeds the 128 MiB preview decode limit."))
                result = np.array(plane, dtype=result_dtype)
            elif z_mode == "max":
                np.maximum(result, plane, out=result)
            else:
                np.add(result, plane, out=result)
            count += 1
        return result / count if z_mode == "mean" else result


def _overview_raw_planes(field, channels, *, time_index=None, z_mode="plane",
                         z_index=None, edge=None, region=None):
    """Yield selected raw planes, validating declared axes and file channel IDs."""
    from PIL import Image

    info = _overview_channel_info(field)
    import operator
    try:
        channels = tuple(operator.index(channel) for channel in channels)
    except TypeError as exc:
        raise _OverviewValidationError(tr("Choose one or more nonnegative channel indices.")) from exc
    for channel in channels:
        if channel < 0:
            raise _OverviewValidationError(tr("Channel indices must be zero or greater."))
        if channel not in info["channels"]:
            raise _OverviewValidationError(tr(
                "Channel {channel} is missing or ambiguous. Available channels: {available}.",
                channel=channel, available=", ".join(map(str, info["channels"])) or "—"))
    if z_mode not in ("plane", "max", "mean"):
        raise _OverviewValidationError(tr("Choose an explicit Z plane or a maximum or mean projection."))
    decoded_path, decoded_array = None, None
    page_cache = {}
    for channel in channels:
        record = (next(record for record in info["files"]
                       if str(record["file_channel"]).isdigit()
                       and int(record["file_channel"]) == channel)
                  if info["kind"] == "files" else info["files"][0])
        path, axes = record["path"], record["axes"]
        selections = {}
        for axis, chosen in (("T", time_index), ("Z", z_index)):
            if axis not in axes and chosen not in (None, 0) and (axis != "Z" or z_mode == "plane"):
                raise _OverviewValidationError(tr("This channel has no image at the selected time or Z plane."))
            if axis not in axes or (axis == "Z" and z_mode != "plane"):
                continue
            if chosen is None:
                raise _OverviewValidationError(tr("Choose explicit time and Z positions before reading this image."))
            count = record["shape"][axes.index(axis)]
            try:
                chosen = operator.index(chosen)
            except TypeError as exc:
                raise _OverviewValidationError(tr("Choose explicit time and Z positions before reading this image.")) from exc
            if not 0 <= chosen < count:
                raise _OverviewValidationError(tr("Axis {axis} index {index} is unavailable; valid indices are 0 to {maximum}.",
                                    axis=axis, index=chosen, maximum=count - 1))
            selections[axis] = int(chosen)
        if path != decoded_path:
            suffix = Path(path).suffix.lower()
            if suffix == ".npy":
                array = np.load(path, mmap_mode="r", allow_pickle=False)
            elif suffix in (".tif", ".tiff"):
                plane_channel = channel if info["kind"] == "stack" else 0
                array = _overview_tiff_plane(record, plane_channel, selections, z_mode, edge, region, page_cache)
                yield channel, path, array
                continue
            else:
                with Image.open(path) as image:
                    if image.width * image.height > 32 * 1024 ** 2:
                        raise _OverviewValidationError(tr("This image exceeds the preview decode limit."))
                    array = np.asarray(image)
            decoded_path, decoded_array = path, array
        array = decoded_array
        selector = tuple(channel if axis == "C" and info["kind"] == "stack"
                         else 0 if axis == "C" else selections.get(axis, slice(None))
                         for axis in axes)
        array = _overview_plane_region(array[selector], region)
        if edge is not None:
            stride = max(1, math.ceil(max(record["shape"][axes.index(axis)] for axis in "YX") / max(1, edge)))
            retained = "".join(axis for axis in axes if axis not in selections and axis != "C")
            array = array[tuple(slice(None, None, stride) if axis in "YX" else slice(None)
                                for axis in retained)]
        if "Z" in axes and z_mode != "plane":
            retained = "".join(axis for axis in axes if axis not in selections and axis != "C")
            stack = np.moveaxis(array, retained.index("Z"), 0)
            result = None
            for index, plane in enumerate(stack):
                if plane.nbytes > 128 * 1024 ** 2:
                    raise _OverviewValidationError(tr("This image exceeds the 128 MiB preview decode limit."))
                if result is None:
                    result = np.array(plane, dtype=np.float64 if z_mode == "mean" else plane.dtype, copy=True)
                elif z_mode == "mean":
                    result += plane
                else:
                    np.maximum(result, plane, out=result)
            array = result / len(stack) if z_mode == "mean" else result
        plane_size = math.prod(record["shape"][axes.index(axis)] for axis in "YX") * array.dtype.itemsize
        if region is None and edge is None and plane_size > 128 * 1024 ** 2:
            raise _OverviewValidationError(tr("This image exceeds the 128 MiB preview decode limit."))
        if array.nbytes > 128 * 1024 ** 2:
            raise _OverviewValidationError(tr("This image exceeds the 128 MiB preview decode limit."))
        if edge is not None:
            stride = max(1, math.ceil(max(array.shape) / max(1, edge)))
            array = array[::stride, ::stride]
        yield channel, path, array


def _overview_channel_rgb(channels, channel_colors):
    """Return one ``(r, g, b)`` weight triple per channel, or None for plain RGB order."""
    if not channel_colors:
        return None
    weights = []
    for channel in channels:
        value = channel_colors.get(channel, channel_colors.get(str(channel)))
        text = str(value or "").lstrip("#")
        if len(text) != 6:
            return None
        try:
            weights.append(tuple(int(text[index:index + 2], 16) / 255. for index in (0, 2, 4)))
        except ValueError:
            return None
    return weights


_OVERVIEW_INKS = {"cyan": (0., 1., 1.), "magenta": (1., 0., 1.), "yellow": (1., 1., 0.), "black": (0., 0., 0.)}
_OVERVIEW_INK_NAMES = {"#00ffff": "cyan", "#ff00ff": "magenta", "#ffff00": "yellow", "#000000": "black"}
_OVERVIEW_INK_SETS = {"cmy": ("cyan", "magenta", "yellow"), "cmyk": ("cyan", "magenta", "yellow", "black")}
_OVERVIEW_COLOR_MODES = {
    "additive": ("rgb", "black"), "rgb_white": ("rgb", "white"),
    "cmy_black": ("cmy", "black"), "cmy": ("cmy", "white"),
    "cmyk_black": ("cmyk", "black"), "cmyk": ("cmyk", "white"),
    "green_magenta": ("green_magenta", "black"), "greys": ("greys", "black"),
    "custom": ("custom", "black"),
}
_OVERVIEW_COLOR_MODE_CAPTIONS = (
    ("additive", "RGB on black"), ("rgb_white", "RGB on white"),
    ("cmy_black", "CMY on black"), ("cmy", "CMY on white"),
    ("cmyk_black", "CMYK on black"), ("cmyk", "CMYK on white"),
    ("green_magenta", "Green and magenta on black"),
    ("greys", "Each channel in grey, side by side"),
    ("custom", "Custom colours"),
)
_OVERVIEW_PRESET_PALETTES = {"rgb": ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
                             "green_magenta": ((0., 1., 0.), (1., 0., 1.), (0., 1., 1.))}
_OVERVIEW_CUSTOM_DEFAULTS = ("#ff0000", "#00ff00", "#0000ff", "#ff00ff", "#00ffff", "#ffff00",
                             "#ff8000", "#8000ff", "#ffffff")


def _overview_inks(channels, channel_colors, mode):
    """The ink each channel prints with in a CMY or CMYK colour mode.

    A channel whose display colour is one of the mode's inks keeps it; the
    others take the unused inks in cyan, magenta, yellow, black order.

    :param channels: the shown channels, in order.
    :param channel_colors: ``{channel: '#rrggbb'}``, or ``None``.
    :param mode: a key of :data:`_OVERVIEW_COLOR_MODES`.
    :returns: one ``(r, g, b)`` ink per channel, or ``None`` outside the CMY
        and CMYK modes or when there are more channels than inks.
    """
    palette = _OVERVIEW_COLOR_MODES.get(str(mode or "additive"), ("rgb", "black"))[0]
    inks = _OVERVIEW_INK_SETS.get(palette, ())
    if not inks or len(channels) > len(inks):
        return None
    colours = channel_colors or {}
    chosen = []
    for channel in channels:
        value = colours.get(channel, colours.get(str(channel)))
        ink = _OVERVIEW_INK_NAMES.get(str(value or "").lower())
        chosen.append(ink if ink in inks and ink not in chosen else None)
    free = [ink for ink in inks if ink not in chosen]
    return [_OVERVIEW_INKS[ink or free.pop(0)] for ink in chosen]


def _overview_hex_rgb(value):
    """``'#rrggbb'`` as an ``(r, g, b)`` triple in 0-1, or ``None`` when it is not one."""
    text = str(value or "").strip().lstrip("#")
    if len(text) != 6:
        return None
    try:
        return tuple(int(text[index:index + 2], 16) / 255. for index in (0, 2, 4))
    except ValueError:
        return None


def _overview_custom_colors(value):
    """Read the ``overview_custom_colors`` setting.

    :param value: ``'1=#ff0000, 2=#00ff00'``, or a mapping.
    :returns: ``{channel text: '#rrggbb'}``; malformed entries are left out.
    """
    if isinstance(value, dict):
        items = value.items()
    else:
        items = (part.split("=", 1) for part in str(value or "").split(",") if "=" in part)
    colours = {}
    for channel, colour in items:
        channel, colour = str(channel).strip(), str(colour).strip().lower()
        if channel and _overview_hex_rgb(colour) is not None:
            colours[channel] = "#" + colour.lstrip("#")
    return colours


def _overview_custom_text(colours):
    """Write ``{channel: '#rrggbb'}`` as the ``overview_custom_colors`` setting."""
    return ", ".join(f"{channel}={colour}" for channel, colour in colours.items())


def _overview_custom_color(colours, channel, index):
    """The custom colour of ``channel``: its own, or the default for its position."""
    own = (colours or {}).get(str(channel))
    return own or _OVERVIEW_CUSTOM_DEFAULTS[index % len(_OVERVIEW_CUSTOM_DEFAULTS)]


def _overview_color_plan(channels, channel_colors, mode):
    """How the shown channels are combined: one colour per channel and a background.

    ``'additive'`` keeps the plain behaviour: the channels' display colours,
    or red, green and blue in order, added on black, one channel in grey and
    more than three side by side. The RGB, CMY and CMYK modes come on black
    (light added) and on white (the complement absorbed, as inks on paper);
    on black the key (black) channel of CMYK is drawn white. ``'custom'``
    uses ``channel_colors`` as chosen per channel, for any number of channels.

    :param channels: the shown channels, in order.
    :param channel_colors: ``{channel: '#rrggbb'}``, or ``None``.
    :param mode: a key of :data:`_OVERVIEW_COLOR_MODES`; anything else is additive.
    :returns: ``(colours, background, inked)`` with one ``(r, g, b)`` in 0-1
        per channel, ``'black'`` or ``'white'``, and whether the ink
        intensities apply; ``None`` when the channels are drawn side by side.
    """
    palette, background = _OVERVIEW_COLOR_MODES.get(str(mode or "additive"), ("rgb", "black"))
    count = len(channels)
    if palette in _OVERVIEW_INK_SETS:
        inks = _overview_inks(channels, channel_colors, mode)
        if inks is None:
            return None
        if background == "black":
            inks = [(1., 1., 1.) if ink == _OVERVIEW_INKS["black"] else ink for ink in inks]
        return inks, background, True
    if palette == "custom":
        colours = {str(channel): value for channel, value in (channel_colors or {}).items()}
        return [_overview_hex_rgb(_overview_custom_color(colours, channel, index))
                for index, channel in enumerate(channels)], background, False
    if palette == "greys":
        return ([(1., 1., 1.)], background, False) if count == 1 else None
    if count > 3:
        return None
    if count == 1 and palette == "rgb":
        return [(1., 1., 1.)], background, False
    weights = _overview_channel_rgb(channels, channel_colors) if palette == "rgb" else None
    return list(weights or _OVERVIEW_PRESET_PALETTES[palette][:count]), background, False


def _overview_composite(planes, colours, background, strengths=None):
    """Combine display planes in their colours on a black or white background.

    On black each channel adds its colour: ``out = sum(plane * strength *
    colour)``. On white it absorbs the complement of its colour from white,
    as an ink on paper: ``out = 255 - sum(plane * strength * (1 - colour))``.
    Both are clipped to 0-255.

    :param planes: uint8 display planes of one shape.
    :param colours: one ``(r, g, b)`` in 0-1 per plane.
    :param background: ``'black'`` or ``'white'``.
    :param strengths: one intensity per plane, or ``None`` for 1 each.
    :returns: the uint8 RGB image.
    """
    if background == "white":
        return _overview_subtractive(planes, colours, strengths)
    mixed = np.zeros(planes[0].shape + (3,), dtype=np.float32)
    for plane, colour, strength in zip(planes, colours, strengths or (1.,) * len(planes)):
        mixed += plane[..., None].astype(np.float32) * float(strength) * np.asarray(colour, dtype=np.float32)
    return np.clip(mixed, 0, 255).astype(np.uint8)


def _overview_subtractive(planes, inks, strengths=None):
    """Print display planes as inks on white paper.

    Each channel absorbs the complement of its ink from white:
    ``out = 255 - sum(plane * strength * (1 - ink))``, clipped to 0-255, so no
    signal is white, one full channel is its ink and full cyan, magenta and
    yellow together are black; a black (key) ink darkens every component
    equally. A channel's strength (its ink intensity) scales how much ink it
    lays down; 1 is the plain print.

    :param planes: uint8 display planes of one shape.
    :param inks: one ``(r, g, b)`` ink in 0-1 per plane.
    :param strengths: one ink intensity per plane, or ``None`` for 1 each.
    :returns: the uint8 RGB print.
    """
    absorbed = np.zeros(planes[0].shape + (3,), dtype=np.float32)
    for plane, ink, strength in zip(planes, inks, strengths or (1.,) * len(planes)):
        absorbed += (plane[..., None].astype(np.float32) * float(strength)
                     * (1. - np.asarray(ink, dtype=np.float32)))
    return np.clip(255. - absorbed, 0, 255).astype(np.uint8)


_OVERVIEW_READOUT_INTERVAL_MS = 33


def _overview_source_pixel(u, v, rendered, source, count=1, mosaic=False):
    """Map a point on a rendered Overview image back to its original source pixel.

    A composite is the source plane sampled every ``stride`` pixels, so a
    rendered pixel covers ``source / rendered`` source pixels in each
    direction. A mosaic (more than three channels side by side) lays the
    channels out as :func:`_overview_pixels` does: square tiles, a caption
    strip on top of each, and one downsampled plane per tile.

    :param u: horizontal position on the rendered image, in rendered pixels.
    :param v: vertical position on the rendered image, in rendered pixels.
    :param rendered: ``(height, width)`` of the rendered image.
    :param source: ``(height, width)`` of the original channel planes.
    :param count: the number of channels drawn.
    :param mosaic: whether the channels are drawn as a mosaic.
    :returns: ``(x, y)`` in original pixels, or ``None`` off the image.
    """
    rows, columns = rendered
    height, width = source
    if mosaic:
        across = math.ceil(math.sqrt(count))
        tile = max(1, columns // across)
        header = min(16, tile // 4)
        column, row = int(u // tile), int(v // tile)
        if u < 0 or v < 0 or column >= across or row * across + column >= count:
            return None
        u, v = u - column * tile, v - row * tile - header
        stride = max(1, math.ceil(max(height, width) / max(1, tile - header)))
        rows, columns = math.ceil(height / stride), math.ceil(width / stride)
    if not (0 <= u < columns and 0 <= v < rows):
        return None
    return min(width - 1, int(u * width / columns)), min(height - 1, int(v * height / rows))


def _overview_readout_planes(field, axes):
    """Read every channel of one field at original resolution for the hover readout.

    Runs off the GUI thread; the planes come from the shared plane cache when
    the views already read them.

    :param field: the hovered field.
    :param axes: ``time_index``, ``z_mode`` and ``z_index``.
    :returns: ``{channel: raw plane}``, or the reason they cannot be read.
    """
    try:
        channels = tuple(_overview_channel_info(field)["channels"])
        return {channel: array for channel, _path, array
                in _overview_cached_raw_planes(field, channels, edge=None, **axes)}
    except Exception as error:
        return tr("Pixel values unavailable: {error}", error=error)


def _overview_raw_value(value):
    """Show one stored pixel value exactly: integers whole, floats to six significant digits."""
    values = np.ravel(np.asarray(value))
    texts = []
    for item in values:
        if np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.bool_):
            texts.append(str(int(item)))
        else:
            texts.append(format(float(item), ".6g"))
    return texts[0] if len(texts) == 1 else "(" + ", ".join(texts) + ")"


def _overview_pixels(field, channels=(0,), percentiles=(1., 99.), edge=1200, *,
                     normalization=None, time_index=None, z_mode="plane", z_index=None,
                     channel_views=False, normalization_context=None, cancelled=None,
                     advanced_normalization=None, region=None, enhancement=None, channel_colors=None,
                     enhancement_channels=None, percentile_channels=None, color_mode="additive",
                     channel_limits=None, image_overrides=None, channel_percentiles=None,
                     channel_enhancement=None, ink_intensity=None):
    """Decode canonical display pixels, optionally returning individual channel views.

    ``enhancement`` is a Make Masks :class:`spacr.qt.detect_chain.Chain`
    applied to each displayed plane after display scaling; it changes only
    what is shown, never raw or measured pixels. ``enhancement_channels`` and
    ``percentile_channels`` restrict the enhancement and the percentile
    stretch to those channels; the others are drawn unenhanced and stretched
    between their minimum and maximum.

    Up to three channels are composited additively in ``channel_colors``
    (``{channel: '#rrggbb'}``); without colours they fill red, green and blue
    in order. ``color_mode`` picks another combination
    (:func:`_overview_color_plan`): RGB, CMY or CMYK on a black or a white
    background, green and magenta, every channel in grey side by side, or
    the custom colours in ``channel_colors``.

    Each channel can be treated on its own: ``channel_percentiles``,
    ``{channel: (low, high)}``, replaces the percentile stretch for that
    channel; ``channel_enhancement``, ``{channel: chain or None}``, replaces
    the enhancement chain for that channel (``None`` shows it unenhanced);
    ``ink_intensity``, ``{channel: strength}``, scales each channel's ink in
    the CMY and CMYK modes. A channel missing from a mapping uses the global
    value, so leaving them out is the plain behaviour.

    Display stages, in order: the global normalization above; then
    ``channel_limits``, ``{image key: {channel: (low, high)}}`` in raw
    intensity units, which replaces a channel's bounds for that image (the
    normalization scope); then ``image_overrides``, ``{image key:
    override}``, applied to the display planes by
    :func:`_overview_override_plane` and to the composite by
    :func:`_overview_saturated`. Both are keyed by :func:`_overview_image_key`.
    An override with ``global_normalization`` False skips the global
    normalization and the scope limits for that image, which is then
    stretched between its minimum and maximum; one with
    ``global_enhancement`` False skips the enhancement chains.
    """
    from PIL import Image, ImageDraw

    planes = []
    plan = _overview_color_plan(channels, channel_colors, color_mode)
    mosaic = plan is None and len(channels) > 1
    columns = math.ceil(math.sqrt(len(channels))) if mosaic else 1
    source_planes = []
    total_bytes = 0
    axes = dict(time_index=time_index, z_mode=z_mode, z_index=z_index)
    read_edge = None if region is not None else edge
    def read_selected(target, selected, edge=read_edge):
        """Read matching physical planes, optionally cropped in original pixels."""
        if region is None or edge is not None:
            yield from _overview_cached_raw_planes(target, selected, edge=edge, **axes)
            return
        for channel, path, values in _overview_raw_planes(target, selected, edge=edge,
                region=region, **axes):
            yield channel, path, values
    image_key = _overview_image_key(field) if (channel_limits or image_overrides) else None
    override = dict((image_overrides or {}).get(image_key) or {})
    if override.get("global_normalization") is False:
        normalization = normalization_context = advanced_normalization = None
        percentiles, channel_limits, channel_percentiles = None, None, None
    if override.get("global_enhancement") is False:
        enhancement = channel_enhancement = None
    own_percentiles = dict(channel_percentiles or {}) if percentiles is not None else {}
    if region is not None and normalization is None and advanced_normalization is None:
        bounds = {}
        for channel, path, raw in read_selected(field, channels, edge=1024):
            finite = np.asarray(raw)[np.isfinite(raw)]
            if finite.size:
                bounds[channel] = ((float(finite.min()), float(finite.max())) if percentiles is None else
                                   tuple(np.percentile(finite, own_percentiles.get(channel, percentiles))))
        normalization = dict(mode="shared", bounds=bounds, gains={}, percentiles=percentiles)
    for item in read_selected(field, channels):
        total_bytes += item[2].nbytes
        if total_bytes > 128 * 1024 ** 2:
            raise _OverviewValidationError(tr("This image exceeds the 128 MiB preview decode limit."))
        source_planes.append(item)
    limits = dict((channel_limits or {}).get(image_key) or {})
    channel_overrides = override.get("channels") or {}
    raw_planes = {channel: array for channel, _path, array in source_planes if channel in limits}
    if advanced_normalization is not None:
        source_planes = _overview_advanced_planes(field, source_planes, advanced_normalization,
            lambda target, selected: read_selected(target, selected, edge=256), read_selected, cancelled)
    else:
        if normalization_context is not None:
            normalization = normalization_context.plan_for(field, cancelled)
        if normalization is not None:
            if region is not None and normalization.get("mode") == "image":
                bounds = {}
                for channel, path, raw in read_selected(field, channels, edge=1024):
                    finite = np.asarray(raw)[np.isfinite(raw)]
                    if finite.size:
                        bounds[channel] = tuple(np.percentile(finite, normalization["percentiles"]))
                reference = normalization.get("reference_channel")
                if reference is not None and reference in bounds:
                    bounds = {channel: bounds[reference] for channel in channels}
                normalization = dict(normalization, mode="shared", bounds=bounds)
            source_planes = tuple(_overview_normalized_planes(source_planes, normalization, field))
    if edge is None:
        edge = (max((max(array.shape) for _, _, array in source_planes), default=1)
                + (16 if mosaic else 0)) * columns
    tile = max(1, edge // columns)
    header = min(16, tile // 4) if mosaic else 0
    plane_edge = max(1, tile - header)
    for channel, path, array in source_planes:
        array = raw_planes.get(channel, array)
        stride = max(1, math.ceil(max(array.shape) / plane_edge))
        plane = np.array(array[::stride, ::stride], dtype=np.float32, copy=True)
        finite = np.isfinite(plane)
        if not finite.any():
            raise _OverviewValidationError(tr("This channel has no finite pixels."))
        if channel in limits:
            low, high = (float(value) for value in limits[channel])
        elif normalization is not None or advanced_normalization is not None:
            low, high = 0., 1.
        elif percentiles is None or (percentile_channels is not None and channel not in percentile_channels):
            low, high = float(plane[finite].min()), float(plane[finite].max())
        else:
            low, high = np.percentile(plane[finite], own_percentiles.get(channel, percentiles))
        plane = np.nan_to_num((plane - low) / max(float(high - low), np.finfo(np.float32).eps),
                              nan=0., posinf=1., neginf=0.)
        chain = (channel_enhancement[channel] if channel_enhancement and channel in channel_enhancement
                 else enhancement)
        if chain is not None and (enhancement_channels is None or channel in enhancement_channels):
            plane = _overview_enhanced_plane(plane, chain, stride, cancelled)
        plane = _overview_override_plane(plane, channel_overrides.get(channel, channel_overrides.get(str(channel))))
        planes.append(np.uint8(np.clip(plane, 0, 1) * 255))
    if not planes or any(plane.shape != planes[0].shape for plane in planes):
        raise _OverviewValidationError(tr("Selected channels must have matching image sizes."))
    if channel_views:
        return tuple((channel, np.repeat(plane[..., None], 3, axis=2))
                     for channel, plane in zip(channels, planes))
    saturation = override.get("saturation")
    if plan is not None:
        colours, background, inked = plan
        strengths = ([float((ink_intensity or {}).get(channel, 1.)) for channel in channels]
                     if inked else None)
        if len(planes) == 1 and colours[0] == (1., 1., 1.) and not inked:
            grey = np.repeat(planes[0][..., None], 3, axis=2)
            return grey if background == "black" else 255 - grey
        return _overview_saturated(_overview_composite(planes, colours, background, strengths), saturation)
    if len(planes) == 1:
        return np.repeat(planes[0][..., None], 3, axis=2)
    rows = math.ceil(len(planes) / columns)
    mosaic_image = Image.new("RGB", (columns * tile, rows * tile))
    draw = ImageDraw.Draw(mosaic_image)
    for index, (channel, plane) in enumerate(zip(channels, planes)):
        x, y = (index % columns) * tile, (index // columns) * tile
        mosaic_image.paste(Image.fromarray(plane).convert("RGB"), (x, y + header))
        if header:
            draw.text((x, y), str(channel), fill="white")
    return np.asarray(mosaic_image).copy()


def _overview_scaled_chain(chain, stride):
    """Express a chain's pixel radii in the pixels of a plane downsampled by ``stride``.

    :param chain: the Make Masks enhancement chain, in original pixels.
    :param stride: the plane's downsampling factor.
    :returns: the chain with its spatial sizes divided by the stride.
    """
    scale = max(1, int(stride))
    if scale == 1:
        return chain
    denoise = chain.denoise_strength
    if chain.denoise in ("gaussian", "median", "bilateral"):
        denoise = max(.1, denoise / scale)
    return chain._replace(background_radius=max(1, round(chain.background_radius / scale)),
                          clahe_tile=max(8, round(chain.clahe_tile / scale)),
                          sharpen_radius=max(.1, chain.sharpen_radius / scale),
                          denoise_strength=denoise)


def _overview_enhanced_plane(plane, chain, stride=1, cancelled=None):
    """Apply Make Masks' enhancement chain to one display plane on 0..1.

    :param plane: display intensities on 0..1.
    :param chain: the enhancement chain, its sizes in original pixels.
    :param stride: how much the plane was downsampled from the source.
    :param cancelled: optional event that stops the chain between stages.
    :returns: the enhanced plane, clipped to 0..1.
    """
    from ..detect_chain import prepare

    unit = np.clip(np.asarray(plane, dtype=np.float32), 0, 1)
    enhanced = prepare(unit, _overview_scaled_chain(chain, stride), cancel=cancelled)
    return np.clip(np.nan_to_num(np.asarray(enhanced, dtype=np.float32), nan=0., posinf=1., neginf=0.), 0, 1)


def _overview_heavy_steps(chain):
    """Name the full-resolution steps a chain asks for, for status text.

    :param chain: the chain applied to an Original image.
    :returns: "PSF deconvolution", "Cellpose 3 restoration" or both.
    """
    steps = []
    if chain.psf_operation != "none":
        steps.append(tr("PSF deconvolution") if chain.psf_operation == "deconvolve" else tr("PSF convolution"))
    if chain.restoration:
        steps.append(tr("Cellpose 3 restoration"))
    return tr(" and ").join(steps)


def _overview_chain(settings):
    """Read Make Masks' enhancement chain from ``enhance_*`` settings.

    :param settings: settings holding the ``enhance_*`` keys.
    :returns: the chain, or ``None`` when no step is switched on.
    :raises ValueError: for a value the chain cannot run with.
    """
    from spacr.psf_pipeline import prepare_chain

    return prepare_chain(dict(settings or {}))


def _overview_measurement(field, channels, cancelled=None, *, time_index=None,
                          z_mode="plane", z_index=None, conditions=None,
                          measurements=None, advanced=None):
    """Measure one whole-image ROI; intensity statistics exclude nonfinite values.

    ``measurements`` is the Overview settings mapping (or a group list) that
    selects the image-level groups; ``advanced`` the prepared Advanced
    options from ``spacr.measure._image_advanced_prepare``. Without
    ``advanced`` the Advanced switches in ``measurements`` run offline.
    """
    import json
    from concurrent.futures import CancelledError
    from skimage.measure import regionprops_table
    from ...measure import (_image_advanced_options, _image_advanced_prepare,
                            _image_feature_groups)

    groups = _image_feature_groups(measurements)
    choices = measurements if isinstance(measurements, dict) else {}
    if advanced is None:
        advanced = _image_advanced_options(choices)
        if any(advanced[name] for name in ("embedding", "deep_qc", "filter_bank")):
            advanced = _image_advanced_prepare(advanced, download=False)

    record = dict(plate=field.plate, well=field.well, row=field.row,
                  column=field.column, field=field.field,
                  field_id=field.field_id or field.field,
                  time_id=field.time_id, z_id=field.z_id,
                  time_index=time_index, z_mode=z_mode, z_index=z_index,
                  condition=(conditions or {}).get((field.plate, field.well), ""),
                  metadata_resolved=not field.unresolved,
                  channels=json.dumps(list(channels)),
                  source_files=json.dumps(field.files),
                  roi="whole_image", geometry_units="pixels",
                  intensity_units="raw_stored_pixel_values",
                  intensity_policy="finite_pixels_only",
                  coordinates="zero_based_row_column_bbox_max_exclusive",
                  status="ok", error="")
    shape = None
    collected = {}
    try:
        planes = (item for channel in channels
                  for item in _overview_raw_planes(field, (channel,),
                      time_index=time_index, z_mode=z_mode, z_index=z_index))
        for channel, path, plane in planes:
            if cancelled is not None and cancelled.is_set():
                raise CancelledError()
            if not (np.issubdtype(plane.dtype, np.number)
                    or np.issubdtype(plane.dtype, np.bool_)) or np.iscomplexobj(plane):
                raise ValueError(tr("Measurements require real numeric image pixels."))
            if shape is None:
                shape = plane.shape
                record.update(height_px=shape[0], width_px=shape[1])
                geometry = regionprops_table(np.ones(shape, dtype=np.uint8), properties=(
                    "area", "bbox", "centroid", "axis_major_length",
                    "axis_minor_length", "eccentricity", "extent", "perimeter",
                    "solidity", "equivalent_diameter_area", "orientation"))
                record.update({"roi_" + key.replace("-", "_"): value[0].item()
                               for key, value in geometry.items()})
            elif plane.shape != shape:
                raise ValueError(tr("Selected channels must have matching image sizes."))
            finite = np.isfinite(plane)
            count = int(np.count_nonzero(finite))
            prefix = "channel_" + str(channel) + "_"
            record.update({prefix + "source": str(path),
                           prefix + "dtype": str(plane.dtype),
                           prefix + "finite_pixel_count": count,
                           prefix + "nonfinite_pixel_count": int(plane.size - count)})
            if not count:
                raise ValueError(tr("This channel has no finite pixels."))
            collected[channel] = plane
        if shape is None:
            raise ValueError(tr("Choose one or more nonnegative channel indices."))
        from ...measure import _image_advanced_features, _image_level_features, _image_object_record
        record.update(_image_level_features(collected, {}, groups=groups))
        record.update(_image_object_record(collected, field.files, field.plate, field.well,
                                           field.field, choices))
        if cancelled is not None and cancelled.is_set():
            raise CancelledError()
        record.update(_image_advanced_features(collected, advanced))
        record["image_features"] = ",".join(groups)
        record["advanced_features"] = ",".join(name for name in ("embedding", "deep_qc", "filter_bank")
                                               if advanced.get(name))
        if "objects" in groups:
            record["object_source"] = "otsu_per_channel"
    except CancelledError:
        raise
    except Exception as error:
        record.update(status="error", error=str(error))
        if collected and "intensity" in groups:
            from ...measure import _image_level_features

            try:
                record.update(_image_level_features(collected, {}, groups=("intensity",)))
            except Exception:
                LOG.debug("partial intensity statistics unavailable", exc_info=True)
    return record



def _overview_normalization_group(field, scope, annotations=None):
    """Use source addresses rather than display labels for normalization groups."""
    import json

    if scope == "image":
        value = ("image", repr(field.identity))
    elif scope == "plate":
        value = ("plate", field.plate)
    elif scope == "well":
        value = ("well", field.plate, field.well)
    elif scope == "row":
        value = ("row", field.plate, field.row)
    elif scope == "column":
        value = ("column", field.plate, field.column)
    else:
        metadata = (annotations or {}).get((field.plate, field.well), {})
        if scope not in metadata or metadata[scope] in (None, ""):
            value = ("unannotated", scope, field.plate, field.well)
        else:
            value = ("annotation", scope, metadata[scope])
    return json.dumps(value, sort_keys=True, default=str)

def _overview_channel_normalization(options, channel):
    """Merge a channel override and validate the display transformation."""
    values = dict(options or {})
    overrides = values.get("per_channel", {})
    values.update(overrides.get(channel, overrides.get(str(channel), {})))
    method = values.get("method", "percentile")
    if method not in ("percentile", "minmax", "manual", "zscore", "log", "asinh", "none", "ratio", "difference"):
        raise ValueError(tr("Choose a supported normalization method."))
    baseline = values.get("baseline", 0.)
    if isinstance(baseline, dict):
        baseline = baseline.get(channel, baseline.get(str(channel), 0.))
    gamma = float(values.get("gamma", 1.))
    baseline = float(baseline)
    if not math.isfinite(gamma) or gamma <= 0 or not math.isfinite(baseline):
        raise ValueError(tr("Gamma must be positive and the baseline must be finite."))
    values.update(method=method, baseline=baseline, gamma=gamma)
    if method in ("ratio", "difference"):
        reference = int(values.get("reference_channel", 0))
        epsilon = float(values.get("epsilon", 1e-8))
        if reference < 0 or epsilon <= 0 or not math.isfinite(epsilon):
            raise ValueError(tr("Reference channel must be nonnegative and epsilon must be positive and finite."))
        values.update(reference_channel=reference, epsilon=epsilon)
    if method in ("percentile", "log", "asinh", "ratio", "difference"):
        low, high = float(values.get("low", 1.)), float(values.get("high", 99.))
        if not 0 <= low < high <= 100:
            raise ValueError(tr("Percentiles must satisfy 0 ≤ low < high ≤ 100."))
    if method == "manual":
        low, high = float(values.get("low", 0.)), float(values.get("high", 255.))
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError(tr("Manual limits must be finite and low must be below high."))
    return values

def _overview_transform(values, options):
    """Apply background subtraction and an optional monotonic display transform."""
    values = np.asarray(values, dtype=np.float64) - options["baseline"]
    if options["method"] == "log":
        return np.sign(values) * np.log1p(np.abs(values))
    if options["method"] == "asinh":
        return np.arcsinh(values)
    return values

def _overview_reference_transform(field, channel, original, options, selection, diagnostics=None, read_planes=None):
    """Compare corresponding physical pixels in the same field, time and Z plane.

    Ratio excludes nonfinite references and references whose absolute value is
    at most epsilon. Difference excludes nonfinite references. Invalid pixels
    display black and never contribute to display limits. Baseline subtraction
    follows the reference operation; raw measurements remain independent.
    """
    values = np.asarray(original[selection], dtype=np.float64)
    method = options["method"]
    if method in ("ratio", "difference"):
        reference_channel = options["reference_channel"]
        if reference_channel == channel:
            reference = original
        else:
            reference = next((read_planes or _overview_raw_planes)(field, (reference_channel,)))[2]
        if reference.shape != original.shape:
            raise ValueError(tr("The reference channel must have the same image size."))
        reference = np.asarray(reference[selection], dtype=np.float64)
        invalid = ~np.isfinite(reference)
        near_zero = np.abs(reference) <= options["epsilon"]
        if method == "ratio":
            invalid |= near_zero
            divided = np.full(values.shape, np.nan)
            np.divide(values, reference, out=divided, where=~invalid)
            values = divided
        else:
            values = np.where(invalid, np.nan, values - reference)
        if diagnostics is not None:
            diagnostics.append(dict(field=field.caption, channel=channel,
                reference_channel=reference_channel, method=method,
                sampled_pixels=int(values.size), invalid_reference_count=int(np.count_nonzero(invalid)),
                near_zero_reference_count=int(np.count_nonzero(near_zero))))
    return _overview_transform(values, options)

def _overview_normalization_bounds(samples, options, transformed=False):
    """Derive display limits from finite transformed samples."""
    values = np.asarray(samples) if transformed else _overview_transform(samples, options)
    if not values.size:
        raise ValueError(tr("This channel has no finite pixels after normalization."))
    method = options["method"]
    if method == "manual":
        low, high = float(options.get("low", 0.)), float(options.get("high", 255.))
    elif method == "none":
        low, high = 0., 255.
    elif method == "zscore":
        mean, std = float(np.mean(values)), float(np.std(values))
        low, high = mean - 3 * std, mean + 3 * std
    elif method == "minmax":
        low, high = float(values.min()), float(values.max())
    else:
        low, high = np.percentile(values, (float(options.get("low", 1.)),
                                           float(options.get("high", 99.))))
    return dict(low=float(low), high=float(high), mean=float(np.mean(values)),
                std=float(np.std(values)), sample_count=int(len(values)))

def _overview_advanced_normalization(fields, channels, options, annotations=None, cancelled=None, read_planes=None):
    """Sample all relevant fields with bounded memory and retain reusable limits.

    The result contains JSON-safe options, per-group channel bounds, field group
    identities, and sampling provenance. Raw image measurements never use it.
    """
    from concurrent.futures import CancelledError

    options = dict(options or {})
    scope = options.get("scope", "image")
    sample_limit = min(65536, max(64, int(options.get("sample_limit", 4096))))
    group_limit = min(1048576, max(sample_limit, int(options.get("group_sample_limit", 262144))))
    total_limit = min(16777216, max(64, int(options.get("total_sample_limit", 4194304))))
    per_group_limit = min(group_limit, total_limit)
    by_channel = {channel: _overview_channel_normalization(options, channel) for channel in channels}
    samples, groups, errors, diagnostics, sampled = {}, {}, [], [], 0
    for field in sorted(fields, key=lambda item: repr(item.identity)):
        if cancelled is not None and cancelled.is_set():
            raise CancelledError()
        group = _overview_normalization_group(field, scope, annotations)
        groups[repr(field.identity)] = group
        try:
            for channel, path, array in (read_planes or _overview_raw_planes)(field, channels):
                if cancelled is not None and cancelled.is_set():
                    raise CancelledError()
                stride = max(1, math.ceil(math.sqrt(array.size / sample_limit)))
                values = _overview_reference_transform(field, channel, array, by_channel[channel],
                    (slice(None, None, stride), slice(None, None, stride)), diagnostics, read_planes).reshape(-1)
                values = values[np.isfinite(values)]
                if not values.size:
                    raise ValueError(tr("This channel has no finite pixels."))
                key = (group, str(channel))
                if key not in samples and (len(samples) + 1) * per_group_limit > total_limit:
                    count = 2 ** math.ceil(math.log2(len(samples) + 1))
                    per_group_limit = max(1, min(group_limit, total_limit // count))
                    for existing, values_so_far in tuple(samples.items()):
                        if len(values_so_far) > per_group_limit:
                            samples[existing] = values_so_far[np.linspace(0, len(values_so_far) - 1, per_group_limit, dtype=int)]
                merged = np.concatenate((samples.get(key, np.empty(0)), values))
                if len(merged) > per_group_limit:
                    merged = merged[np.linspace(0, len(merged) - 1, per_group_limit, dtype=int)]
                samples[key] = merged
            sampled += 1
        except CancelledError:
            raise
        except Exception as error:
            errors.append((field.caption, str(error)))
    bounds = {}
    for (group, channel), values in samples.items():
        bounds.setdefault(group, {})[channel] = _overview_normalization_bounds(
            values, by_channel[int(channel)], transformed=True)
    return dict(options=options, bounds=bounds, field_groups=groups,
                provenance=dict(sample_limit=sample_limit, group_sample_limit=group_limit,
                                total_sample_limit=total_limit,
                                field_count=len(fields), sampled_fields=sampled, errors=errors,
                                reference_diagnostics=diagnostics,
                                policy="finite transformed pixel samples; display only"))

def _overview_advanced_planes(field, planes, model, sample_reader, reference_reader, cancelled=None):
    """Apply independent channel transforms with shared, source-bound limits."""
    planes = tuple(planes)
    options = model.get("options", model)
    if not model.get("bounds") and options.get("scope", "image") == "image":
        model = _overview_advanced_normalization((field,), tuple(channel for channel, _, _ in planes),
            options, cancelled=cancelled, read_planes=sample_reader)
    group = model.get("field_groups", {}).get(repr(field.identity))
    bounds = model.get("bounds", {}).get(group, {})
    result = []
    for channel, path, raw in planes:
        if cancelled is not None and cancelled.is_set():
            from concurrent.futures import CancelledError
            raise CancelledError()
        settings = _overview_channel_normalization(options, channel)
        limits = bounds.get(str(channel))
        if limits is None:
            raise _OverviewValidationError(tr("No normalization samples are available for this channel."))
        values = _overview_reference_transform(field, channel, raw, settings,
            (slice(None), slice(None)), read_planes=reference_reader)
        scaled = np.nan_to_num((values - limits["low"]) / max(limits["high"] - limits["low"],
            np.finfo(np.float64).eps), nan=0., posinf=1., neginf=0.)
        result.append((channel, path, np.asarray(np.clip(scaled, 0, 1) ** (1. / settings["gamma"]), dtype=np.float32)))
    return tuple(result)


_OVERVIEW_WORKER_START_DELAY = 10.0

_OVERVIEW_ADVANCED_RAM = dict(embedding=1024 ** 3, deep_qc=384 * 1024 ** 2, filter_bank=0)


def _overview_measurement_pool(workers, cancelled=None):
    """Start Measure's paced spawn pool for one-image overview measurement jobs.

    Workers run Measure's single-thread initializer, start
    ``_OVERVIEW_WORKER_START_DELAY`` seconds apart through ``_parallel_pool``
    (which binds them to the parent's Windows job object) and stop waiting to
    start once ``cancelled`` is set.
    """
    import multiprocessing
    from ...measure import _initialize_measure_worker
    from ...resource_log import _WorkerStartGate, _parallel_pool

    gate = _WorkerStartGate(_OVERVIEW_WORKER_START_DELAY)
    if cancelled is not None:
        gate.idle = cancelled.is_set
    return _parallel_pool(workers, initializer=_initialize_measure_worker,
                          context=multiprocessing.get_context("spawn"), gate=gate)


def _overview_worker_ram(field, channels, advanced):
    """Estimate one overview worker's peak memory in bytes, or 0 when unknown."""
    try:
        info = _overview_channel_info(field)
        shape = info["files"][0]["shape"]
        axes = info["files"][0]["axes"]
        pixels = shape[axes.index("Y")] * shape[axes.index("X")]
    except Exception:
        return 0
    overhead = sum(size for name, size in _OVERVIEW_ADVANCED_RAM.items() if advanced.get(name))
    return int(pixels * 8 * len(channels) * 8 + overhead)


def _overview_measurements_export(fields, source, channels, workers=1,
                                  cancelled=None, annotations=None, progress=None, *,
                                  time_index=None, z_mode="plane", z_index=None, field_options=None,
                                  measurements=None):
    """Publish an atomic raw-pixel FOV run, with one bounded image per worker.

    ``progress(completed, total)`` runs in the caller thread. Failed or cancelled
    runs leave existing measurements intact and return per-field diagnostics.
    Selected channels replace the previous row for the same source field.

    Parallel runs use Measure's pool: ``_parallel_pool`` spawn workers with
    single-threaded libraries, at most ``workers`` images in flight, the
    worker count capped by the image count and lowered by the RAM plan
    (``overview_measure_ram_guard``), and each next image held while RAM is
    short. ``workers=None`` takes ``overview_measure_n_jobs`` from
    ``measurements``, resolved like Measure's ``n_jobs``. ``measurements`` is
    the Overview settings mapping choosing the measurement groups and the
    Advanced operations; their models are prepared once here, before any
    worker starts, and embedding principal components are added over the run.
    A finished run closes and joins its workers, as Measure does, so they
    exit on their own; only a cancelled or failed run terminates them.
    """
    import hashlib
    import json
    import sqlite3
    import tempfile
    import time
    from concurrent.futures import CancelledError
    from datetime import datetime, timezone
    from ...measure import (_image_advanced_options, _image_advanced_prepare,
                            _image_embedding_components, _image_feature_groups,
                            _publish_image_measurements,
                            _single_threaded_children, _wait_for_ram,
                            resolve_n_jobs, resolve_pool_size)
    from ...resource_log import _clamp_to_plan, _ram_plan
    from ...schema import row_id, column_id, field_id, time_id, compose_prc, compose_prcf

    source = Path(source).expanduser().resolve()
    destination = source / "measurements" / "measurements.db"
    channels = tuple(dict.fromkeys(int(value) for value in channels))
    if not channels or any(value < 0 for value in channels):
        raise ValueError(tr("Choose one or more nonnegative channel indices."))
    choices = dict(measurements) if isinstance(measurements, dict) else {}
    _image_feature_groups(measurements)
    if workers is None:
        workers = resolve_n_jobs(choices.get("overview_measure_n_jobs"))
    workers = max(1, min(int(workers), os.cpu_count() or 1))
    fields = tuple(fields)
    report = dict(path=str(destination), count=0, errors=[], cancelled=False,
                  attempted=0, diagnostics=[], workers=workers, advanced_unavailable={})
    if not fields:
        return report
    advanced = _image_advanced_options(choices)
    if any(advanced[name] for name in ("embedding", "deep_qc", "filter_bank")):
        advanced = _image_advanced_prepare(advanced)
        report["advanced_unavailable"] = dict(advanced["unavailable"])
    else:
        advanced = dict(advanced, unavailable={})
    if workers > 1:
        workers = resolve_pool_size(workers, len(fields))
        per_worker = _overview_worker_ram(fields[0], channels, advanced)
        plan = _ram_plan(per_worker, workers, module="overview_measure", multiplier=1.0)
        workers = max(1, int(_clamp_to_plan(workers, plan, choices.get("overview_measure_ram_guard", True))))
        report["workers"] = workers
    else:
        per_worker = 0
    annotations = annotations or {}
    measured_at = datetime.now(timezone.utc).isoformat()

    def axes_for(field):
        """Resolve immutable per-selection axes for every worker and database row."""
        return dict(time_index=time_index, z_mode=z_mode, z_index=z_index) | dict((field_options or {}).get(repr(field.identity), {}))

    def check_cancelled():
        """Raise cancellation when the measurement export is stopped."""
        if cancelled is not None and cancelled.is_set():
            raise CancelledError()

    def record_result(field, row, stage):
        """Record one field result or diagnostic in the staged export."""
        check_cancelled()
        report["attempted"] += 1
        if row.get("status") != "ok":
            diagnostic = dict(plate=field.plate, well=field.well, field=field.field,
                              time=str(field.time_id),
                              z=str(field.z_id),
                              source_files=field.files,
                              error=row.get("error", tr("Measurement failed.")))
            report["errors"].append(diagnostic)
            report["diagnostics"].append(diagnostic)
        else:
            info = _overview_channel_info(field)
            has_time = any("T" in item["axes"] for item in info["files"])
            has_z = any("Z" in item["axes"] for item in info["files"])
            axes = axes_for(field)
            selected_time = axes["time_index"] if has_time else None
            selected_mode = axes["z_mode"] if has_z else "plane"
            selected_z = axes["z_index"] if has_z and selected_mode == "plane" else None
            row.update(time_index=selected_time, z_mode=selected_mode, z_index=selected_z)
            identity = [str(source), field.plate, field.well, field.field_id or field.field,
                        str(field.time_id), str(field.z_id),
                        [selected_time, selected_mode, selected_z],
                        sorted((str(key), str(Path(path).resolve()))
                               for key, path in field.files)]
            metadata = annotations.get((field.plate, field.well),
                annotations.get(field.plate + "/" + field.well,
                    annotations.get(field.well, {})))
            physical_field = field.field_id or field.field
            acquisition_time = field.time_id or None
            if acquisition_time is None and has_time:
                acquisition_time = selected_time
            row.update(field_uid=hashlib.sha256(json.dumps(identity).encode()).hexdigest(),
                       source=str(source), measured_at=measured_at,
                       plateID=field.plate, rowID=row_id(field.row),
                       columnID=column_id(field.column), fieldID=field_id(physical_field),
                       timeID=time_id(acquisition_time) if acquisition_time is not None else "",
                       sliceID=str(field.z_id),
                       prc=compose_prc(field.plate, field.row, field.column),
                       prcf=compose_prcf(field.plate, field.row, field.column, physical_field, acquisition_time),
                       condition=str(metadata.get("condition", "")),
                       annotations=json.dumps(metadata, sort_keys=True))
            stage.execute("INSERT INTO records VALUES (?)", (json.dumps(row),))
        if progress is not None:
            progress(report["attempted"], len(fields))

    def measure_kwargs(field):
        """Keyword arguments of one image's measurement job."""
        return dict(axes_for(field), measurements=measurements, advanced=advanced)

    try:
        check_cancelled()
        with tempfile.TemporaryDirectory(prefix="spacr-fov-") as temporary:
            stage = sqlite3.connect(str(Path(temporary) / "stage.db"), timeout=30)
            try:
                stage.execute("CREATE TABLE records (record TEXT NOT NULL)")
                if workers == 1:
                    for field in fields:
                        check_cancelled()
                        record_result(field, _overview_measurement(field, channels, cancelled,
                            **measure_kwargs(field)), stage)
                else:
                    pending = {}
                    remaining = iter(fields)

                    def submit():
                        """Queue the next image once RAM allows; False when none remain."""
                        field = next(remaining, None)
                        if field is None:
                            return False
                        check_cancelled()
                        if per_worker:
                            _wait_for_ram(per_worker, lambda: bool(pending), field=field.field)
                        pending[pool.apply_async(_overview_measurement, (field, channels),
                                                 measure_kwargs(field))] = field
                        return True

                    with _single_threaded_children(), _overview_measurement_pool(workers, cancelled) as pool:
                        while len(pending) < workers and submit():
                            pass
                        while pending:
                            check_cancelled()
                            done = []
                            for result in list(pending):
                                try:
                                    if result.ready():
                                        done.append((result, None))
                                except Exception as error:
                                    done.append((result, error))
                            if not done:
                                time.sleep(.05)
                                continue
                            for result, failure in done:
                                field = pending.pop(result)
                                if failure is None:
                                    try:
                                        row = result.get(timeout=0)
                                    except Exception as error:
                                        failure = error
                                if failure is not None:
                                    row = dict(status="error", error=str(failure))
                                record_result(field, row, stage)
                                submit()
                        pool.close()
                        pool.join()
                check_cancelled()
                if report["errors"]:
                    return report
                check_cancelled()
                rows = [json.loads(value) for (value,) in stage.execute("SELECT record FROM records")]
                if advanced.get("embedding"):
                    components = _image_embedding_components({
                        row["field_uid"]: row.get("embedding_vector") for row in rows})
                    for row in rows:
                        row.update(components.get(row["field_uid"], {}))
                destination.parent.mkdir(parents=True, exist_ok=True)
                published = _publish_image_measurements(destination, rows, choices, cancelled=cancelled)
                report.update(path=str(published.get("path", destination)),
                              exports={key: str(value) for key, value in (published.get("exports") or {}).items()},
                              inserted=published.get("inserted", 0), updated=published.get("updated", 0))
                report["count"] = report["attempted"]
            finally:
                stage.close()
    except CancelledError:
        report["cancelled"] = True
    return report


def _overview_sample(fields, count=1, seed=0, grouping="well", randomize=False):
    """Select stable examples per actual plate, row, column or well."""
    import random
    grouped = {}
    for field in sorted(fields, key=lambda value: value.identity):
        key = (field.plate,)
        if grouping == "row":
            key += (field.row,)
        elif grouping == "column":
            key += (field.column,)
        elif grouping == "well":
            key += (field.well,)
        if field.unresolved:
            key += (field.identity,)
        grouped.setdefault(key, []).append(field)
    rng = random.Random(seed)
    result = []
    for key in sorted(grouped, key=str):
        values = grouped[key]
        n = min(max(1, int(count)), len(values))
        result.extend(rng.sample(values, n) if randomize else values[:n])
    return tuple(result)


def _overview_color_mode_caption(mode):
    """The colour mode's name as the settings list it."""
    captions = dict(_OVERVIEW_COLOR_MODE_CAPTIONS)
    return tr(captions.get(str(mode or "additive"), captions["additive"]))


def _overview_pdf_header(channels, percentiles, seed, normalization, color_mode="additive"):
    """Describe the selected display transform and its actual sampled reference."""
    import textwrap

    lines = [tr("Channels: {channels}; percentiles: {percentiles}; seed: {seed}",
                channels=str(channels), percentiles=str(percentiles), seed=seed)]
    if str(color_mode or "additive") != "additive":
        lines.append(tr("Colour mode: {mode}", mode=_overview_color_mode_caption(color_mode)))
    if normalization is not None:
        modes = {"image": tr("Percentiles per image"),
                 "shared": tr("Shared channel percentile bounds"),
                 "reference_well": tr("Bounds from a reference well"),
                 "reference_row": tr("Bounds from a reference row"),
                 "reference_column": tr("Bounds from a reference column"),
                 "reference_condition": tr("Bounds from a reference condition"),
                 "reference_average": tr("Well-mean display scaling to reference average")}
        reference = normalization.get("reference")
        if isinstance(reference, (tuple, list)):
            reference = " / ".join(map(str, reference))
        reference = str(reference) if reference not in (None, "") else "-"
        reference_channel = normalization.get("reference_channel")
        matching = tr("On") if normalization.get("mean_match") else tr("Off")
        gains = normalization.get("gains") or {}
        if normalization.get("mean_match"):
            matching += " (" + ", ".join(f"{channel} × {float(gains.get(channel, 1.)):g}"
                                         for channel in channels) + ")"
        sample_fields = int(normalization.get("sample_fields", 0))
        counts = normalization.get("sample_pixels") or {}
        if normalization["mode"] == "reference_average":
            reference_wells = normalization.get("reference_wells") or ()
            sample_fields = sum(len(well["field_identities"]) for well in reference_wells)
            counts = {channel: sum(well["sample_pixels"].get(channel, 0)
                                   for well in reference_wells) for channel in channels}
            matching = "; ".join(
                f"{well['plate']} / {well['well']} ("
                + ", ".join(f"{channel} × {float(well['gains'][channel]):g}"
                            for channel in channels) + ")"
                for well in normalization.get("used_wells", ()))
        pixels = ", ".join(f"{channel}: {int(counts.get(channel, 0))}" for channel in channels)
        lines.extend((
            tr("Display normalization") + ": " + modes[normalization["mode"]]
            + " | " + tr("Reference scope") + ": " + reference,
            tr("Use reference-channel bounds") + ": "
            + (str(reference_channel) if reference_channel is not None else "-"),
            tr("Mean-matched display scaling before reference bounds") + ": " + matching,
            tr("Sampled fields: {fields}; sampled pixels per channel: {pixels}",
               fields=sample_fields, pixels=pixels),
        ))
        if normalization["mode"] == "reference_average":
            means = ", ".join(f"{channel}: {float(normalization['reference_means'][channel]):g}"
                              for channel in channels)
            lines.append(tr("Reference means: {means}", means=means))
    return tuple(part for line in lines for part in (textwrap.wrap(line, 112) or [""]))


def _overview_channel_display_record(channels, percentiles, channel_percentiles=None, ink_intensity=None):
    """Each exported channel's own display values, for ``overview_selection.json``.

    :param channels: the exported channels.
    :param percentiles: the global percentile stretch, or ``None``.
    :param channel_percentiles: ``{channel: (low, high)}`` channel overrides.
    :param ink_intensity: ``{channel: strength}`` for the CMY and CMYK modes.
    :returns: ``{channel: {"percentiles": [low, high] or None, "ink_intensity": s}}``.
    """
    record = {}
    for channel in channels:
        own = (channel_percentiles or {}).get(channel) if percentiles is not None else None
        chosen = own if own is not None else percentiles
        record[str(channel)] = {"percentiles": None if chosen is None else [float(value) for value in chosen],
                                "ink_intensity": float((ink_intensity or {}).get(channel, 1.))}
    return record


def _overview_channel_display_lines(channels, channel_percentiles=None, ink_intensity=None,
                                    color_mode="additive"):
    """PDF header lines naming the channels that have their own display values.

    :returns: the wrapped lines; none when every channel uses the global values.
    """
    import textwrap

    parts = []
    for channel in channels:
        values = []
        own = (channel_percentiles or {}).get(channel)
        if own is not None:
            values.append(tr("percentiles {low}-{high}", low=f"{float(own[0]):g}", high=f"{float(own[1]):g}"))
        strength = float((ink_intensity or {}).get(channel, 1.))
        if str(color_mode or "additive") != "additive" and strength != 1.:
            values.append(tr("ink intensity {value}", value=f"{strength:g}"))
        if values:
            parts.append(f"{channel}: " + ", ".join(values))
    if not parts:
        return ()
    return tuple(textwrap.wrap(tr("Channel display") + ": " + "; ".join(parts), 112))


def _overview_pdf_pixels(pixels, axis, dpi):
    """Thin a field's pixels to at most ``dpi`` pixels per printed inch on its page.

    The images are embedded without interpolation, so before this the page
    DPI changed nothing in the file: 150, 300 and 600 wrote the same pixels.
    Pixels are only ever dropped, by a whole-pixel stride, never invented.
    The export applies it to bounded PDF qualities only: "full source
    resolution" keeps every source pixel whatever the DPI.

    :param pixels: the field's image, rows first.
    :param axis: the page axis it is drawn in, already placed on the figure.
    :param dpi: the page resolution in dots per inch, or ``None`` to keep all.
    :returns: the pixels, strided when they print denser than ``dpi``.
    """
    if not dpi:
        return pixels
    width_in, height_in = axis.figure.get_size_inches()
    box = axis.get_position()
    rows, columns = pixels.shape[:2]
    inches = min(box.width * width_in / max(1, columns), box.height * height_in / max(1, rows))
    if inches <= 0:
        return pixels
    longest = max(rows, columns)
    limit = max(1, math.floor(float(dpi) * longest * inches + 1e-9))
    step = math.ceil(longest / limit)
    if step <= 1:
        return pixels
    return np.ascontiguousarray(pixels[::step, ::step])


def _overview_export(fields, folder, channels, percentiles, seed=0, cancelled=None,
                     measurements=False, *, quality_edge=1000, dpi=None,
                     normalization=None, time_index=None, z_mode="plane", z_index=None,
                     conditions=None, normalization_context=None, annotations=None,
                     advanced_normalization=None, field_options=None, channel_colors=None,
                     color_mode="additive", load_limits=None, channel_percentiles=None,
                     ink_intensity=None):
    """Write twelve bounded images per styled PDF with reproducible provenance.

    ``channel_percentiles`` and ``ink_intensity`` are the per-channel display
    values (see :func:`_overview_pixels`); they are applied to the pages and
    recorded in the header and in ``overview_selection.json``.
    """
    import json
    from matplotlib.figure import Figure
    from ...plot import save_figure
    from ...figures.style import _apply_user_style
    from ...figures.bundle import _register_figure_data
    from ... import tabular
    from concurrent.futures import CancelledError

    destination = Path(folder)
    destination.mkdir(parents=True, exist_ok=True)
    if (any(destination.glob("overview_*.pdf"))
            or (destination / "overview_selection.json").exists()
            or (destination / "overview_measurements.csv").exists()):
        raise ValueError(tr("Overview files already exist here. Choose a new output folder."))
    written, errors, provenance = [], [], []
    measurement_rows = []
    axis_options = dict(time_index=time_index, z_mode=z_mode, z_index=z_index)
    for start in range(0, len(fields), 12):
        if cancelled is not None and cancelled.is_set():
            break
        page_fields = fields[start:start + 12]
        columns = min(3, len(page_fields)) if len(page_fields) != 4 else 2
        rows = math.ceil(len(page_fields) / columns)
        if normalization_context is not None:
            try:
                for field in page_fields:
                    normalization_context.plan_for(field, cancelled)
            except CancelledError:
                break
        page_normalization = (normalization_context.provenance()
                              if normalization_context is not None else normalization)
        header = _overview_pdf_header(channels, percentiles, seed,
            None if advanced_normalization is not None else page_normalization, color_mode)
        header = tuple(header) + _overview_channel_display_lines(channels, channel_percentiles,
                                                                 ink_intensity, color_mode)
        if advanced_normalization is not None:
            settings = advanced_normalization.get("options", advanced_normalization)
            header = tuple(header) + (tr("Channels: {channels}; normalization: {method}; scope: {scope}; seed: {seed}",
                channels=str(channels), method=settings.get("method"), scope=settings.get("scope"), seed=seed),)
            transforms = []
            for channel in channels:
                effective = _overview_channel_normalization(settings, channel)
                caption = str(channel) + ": " + effective["method"]
                caption += ", " + tr("Subtract background") + "=" + str(effective["baseline"])
                caption += ", " + tr("Gamma") + "=" + str(effective["gamma"])
                if effective["method"] in ("ratio", "difference"):
                    caption += ", " + tr("Reference channel (ratio / difference)") + "=" + str(effective["reference_channel"])
                    caption += ", epsilon=" + str(effective["epsilon"])
                transforms.append(caption)
            import textwrap
            header += tuple(textwrap.wrap(tr("Channels") + ": " + "; ".join(transforms), 112))
        header_height = .25 + .18 * len(header)
        panel_width = 12. / columns
        import textwrap
        captions = []
        for field in page_fields:
            metadata = (annotations or {}).get((field.plate, field.well), {})
            text = field.caption
            selected_axes = axis_options | dict((field_options or {}).get(repr(field.identity), {}))
            try:
                info = _overview_channel_info(field)
            except (OSError, ValueError):
                info = {"files": ()}
            if any("T" in item["axes"] for item in info["files"]):
                text += " / T=" + str(selected_axes["time_index"])
            if any("Z" in item["axes"] for item in info["files"]):
                mode = selected_axes["z_mode"]
                text += " / " + ("Z=" + str(selected_axes["z_index"]) if mode == "plane" else
                    tr("Maximum Z projection") if mode == "max" else tr("Mean Z projection"))
            if metadata:
                text += " / " + ", ".join(str(key) + ": " + str(value) for key, value in sorted(metadata.items()))
            elif (conditions or {}).get((field.plate, field.well), ""):
                text += " / " + str(conditions[(field.plate, field.well)])
            captions.append("\n".join(textwrap.wrap(text, max(24, int(95 / columns)))))
        heights = []
        for row in range(rows):
            lines = max(caption.count("\n") + 1 for caption in captions[row * columns:(row + 1) * columns])
            heights.extend((.18 * lines + .12, panel_width + .25))
        figure_height = sum(heights) + header_height + .6
        figure = Figure(figsize=(12, figure_height))
        grid = figure.add_gridspec(rows * 2, columns, height_ratios=heights,
            left=.035, right=.965, bottom=.03,
            top=1. - (header_height + .12) / figure_height,
            hspace=.08, wspace=.12)
        figure.text(.5, 1. - .14 / figure_height, "\n".join(header),
                    ha="center", va="top", fontsize=9, linespacing=1.3)
        page_records, page_errors, page_images, image_panels, image_channels = [], [], [], [], []
        interrupted = False
        for index, field in enumerate(page_fields):
            selected_axes = axis_options | dict((field_options or {}).get(repr(field.identity), {}))
            if cancelled is not None and cancelled.is_set():
                interrupted = True
                break
            if measurements:
                try:
                    measurement_rows.append(_overview_measurement(field, channels, cancelled,
                                            conditions=conditions, **selected_axes))
                    if cancelled is not None and cancelled.is_set():
                        raise CancelledError()
                except CancelledError:
                    interrupted = True
                    break
            panel = grid[(index // columns) * 2 + 1, index % columns]
            condition = (conditions or {}).get((field.plate, field.well), "")
            bounds = grid[(index // columns) * 2, index % columns].get_position(figure)
            figure.text((bounds.x0 + bounds.x1) / 2, bounds.y1 - .05 / figure_height,
                        captions[index], ha="center", va="top", fontsize=10,
                        linespacing=1.25)
            try:
                if len(channels) > 3:
                    views = _overview_pixels(field, channels, percentiles, edge=quality_edge,
                        normalization=normalization, normalization_context=normalization_context,
                        cancelled=cancelled, channel_percentiles=channel_percentiles,
                        channel_views=True, advanced_normalization=advanced_normalization, **selected_axes)
                    channel_columns = math.ceil(math.sqrt(len(views)))
                    channel_rows = math.ceil(len(views) / channel_columns)
                    inner = panel.subgridspec(channel_rows, channel_columns, hspace=.18, wspace=.08)
                    for position, (channel, pixels) in enumerate(views):
                        axis = figure.add_subplot(inner[position // channel_columns, position % channel_columns])
                        pixels = _overview_pdf_pixels(pixels, axis, None if quality_edge is None else dpi)
                        axis.imshow(pixels, interpolation="none")
                        axis.set_title(str(channel), fontsize=10, pad=5)
                        axis.set_axis_off()
                        page_images.append(pixels)
                        image_panels.append(index)
                        image_channels.append((channel,))
                else:
                    pixels = _overview_pixels(field, channels, percentiles, edge=quality_edge,
                                              normalization=normalization,
                                              normalization_context=normalization_context,
                                              cancelled=cancelled, advanced_normalization=advanced_normalization,
                                              channel_colors=channel_colors, color_mode=color_mode,
                                              channel_percentiles=channel_percentiles,
                                              ink_intensity=ink_intensity, **selected_axes)
                    axis = figure.add_subplot(panel)
                    pixels = _overview_pdf_pixels(pixels, axis, None if quality_edge is None else dpi)
                    axis.imshow(pixels, interpolation="none")
                    axis.set_axis_off()
                    page_images.append(pixels)
                    image_panels.append(index)
                    image_channels.append(tuple(channels))
            except CancelledError:
                interrupted = True
                break
            except Exception as exc:
                axis = figure.add_subplot(panel)
                axis.text(.5, .5, str(exc), ha="center", va="center", wrap=True)
                axis.set_axis_off()
                page_errors.append((field.caption, str(exc)))
            page_records.append({"plate": field.plate, "well": field.well,
                               "field": field.field, "files": field.files,
                               "field_id": field.field_id or field.field,
                               "time_id": field.time_id, "z_id": field.z_id,
                               "condition": condition, **selected_axes,
                               "annotations": (annotations or {}).get((field.plate, field.well), {})})
        if interrupted:
            figure.clear()
            break
        _register_figure_data(figure, page_images, kind="montage", fields=page_records,
                              channels=channels, percentiles=percentiles, seed=seed,
                              image_panels=image_panels, image_channels=image_channels,
                              errors=page_errors, grid=(rows, columns), preview_edge=quality_edge,
                              original_resolution=quality_edge is None,
                              advanced_normalization=advanced_normalization,
                              normalization=(normalization_context.provenance()
                                  if normalization_context is not None else normalization), **axis_options)
        _apply_user_style(figure)
        path = destination / f"overview_{start // 12 + 1:04d}.pdf"
        save_figure(figure, str(path), fmt="pdf", close=True, dpi=dpi)
        written.append(str(path))
        provenance.extend(page_records)
        errors.extend(page_errors)
    stopped = bool(cancelled is not None and cancelled.is_set())
    csv_path = None
    if measurements and not stopped:
        import pandas as pd
        csv_path = str(destination / "overview_measurements.csv")
        def _write_measurements(pending):
            """Write pending CSV and refuse publication after cancellation."""
            tabular.write_table(pd.DataFrame(measurement_rows), pending, canonicalise=False)
            if cancelled is not None and cancelled.is_set():
                raise CancelledError()
        try:
            tabular._publish(csv_path, _write_measurements)
        except CancelledError:
            stopped, csv_path = True, None
        known = {caption for caption, message in errors}
        errors.extend((field.caption, row["error"]) for field, row in zip(fields, measurement_rows)
                      if row["status"] == "error" and field.caption not in known)
    metadata = {"channels": channels, "percentiles": percentiles, "seed": seed,
                "field_options": field_options, "color_mode": str(color_mode or "additive"),
                "load_limits": {"max_wells_per_plate": (load_limits or (None, None))[0],
                                "max_fields_per_well": (load_limits or (None, None))[1]},
                "channel_display": _overview_channel_display_record(channels, percentiles,
                                                                    channel_percentiles, ink_intensity),
                "fields": provenance, "errors": errors, "cancelled": stopped,
                "requested_field_count": len(fields), "written_field_count": len(provenance),
                "quality_edge": quality_edge, "dpi": dpi,
                "advanced_normalization": advanced_normalization,
                "normalization": (normalization_context.provenance()
                    if normalization_context is not None else normalization),
                **axis_options}
    if measurements:
        metadata["measurements"] = dict(path=csv_path, roi="whole_image",
            geometry_units="pixels", intensity_units="raw_stored_pixel_values",
            intensity_policy="finite_pixels_only", original_resolution=True,
            coordinates="zero_based_row_column_bbox_max_exclusive",
            written_field_count=len(measurement_rows) if csv_path else 0,
            cancelled=stopped)
    tabular._publish(str(destination / "overview_selection.json"), lambda pending:
        Path(pending).write_text(json.dumps(metadata, indent=2), encoding="utf-8"))
    return tuple(written), tuple(errors)


def _overview_montage(fields, channels, percentiles, cancelled=None, *,
                      normalization=None, time_index=None, z_mode="plane", z_index=None,
                      normalization_context=None, advanced_normalization=None, preview_edge=240,
                      labels=True, enhancement=None, channel_colors=None, enhancement_channels=None,
                      percentile_channels=None, color_mode="additive", channel_limits=None,
                      image_overrides=None, channel_percentiles=None, channel_enhancement=None,
                      ink_intensity=None):
    """Render one bounded page of field thumbnails, using the shared pixel path.

    ``labels`` draws each tile's well and field name above its image.
    """
    from concurrent.futures import CancelledError
    from PIL import Image, ImageDraw

    columns = min(4, max(1, len(fields)))
    rows = max(1, math.ceil(len(fields) / columns))
    image = Image.new("RGB", (columns * 256, rows * 280))
    draw = ImageDraw.Draw(image)
    errors = []
    for index, field in enumerate(fields):
        if cancelled is not None and cancelled.is_set():
            return None, ()
        x, y = (index % columns) * 256, (index // columns) * 280
        if labels:
            draw.text((x, y), field.well + " / " + field.field, fill="white")
        try:
            pixels = _overview_pixels(field, channels, percentiles, edge=preview_edge,
                                      normalization=normalization,
                                      normalization_context=normalization_context,
                                      cancelled=cancelled, time_index=time_index,
                                      z_mode=z_mode, z_index=z_index,
                                      advanced_normalization=advanced_normalization,
                                      enhancement=enhancement, channel_colors=channel_colors,
                                      enhancement_channels=enhancement_channels,
                                      percentile_channels=percentile_channels, color_mode=color_mode,
                                      channel_limits=channel_limits, image_overrides=image_overrides,
                                      channel_percentiles=channel_percentiles,
                                      channel_enhancement=channel_enhancement, ink_intensity=ink_intensity)
            preview = Image.fromarray(pixels)
            factor = min(240 / preview.width, 240 / preview.height)
            preview = preview.resize((max(1, round(preview.width * factor)),
                                      max(1, round(preview.height * factor))), Image.Resampling.BILINEAR)
            image.paste(preview, (x, y + 24))
        except CancelledError:
            return None, ()
        except Exception as exc:
            errors.append((field.caption, str(exc)))
            draw.text((x, y + 30), str(exc), fill="red")
    return np.asarray(image).copy(), tuple(errors)


class _OverviewPlateDelegate(QStyledItemDelegate):
    """Show the active physical well beneath a translucent blue hover."""

    def paint(self, painter, option, index):
        """Reserve native image space and keep the complete caption in its tooltip.

        :param painter: the active GUI-thread painter.
        :param option: native cell geometry and hover state.
        :param index: the physical well cell.
        """
        opts = QStyleOptionViewItem(option)
        self.initStyleOption(opts, index)
        collapsed = getattr(option.widget, "_overview_collapsed", None)
        if collapsed and (index.row() in collapsed[0] or index.column() in collapsed[1]):
            strip = QColor(opts.palette.color(QPalette.Highlight))
            strip.setAlpha(110)
            painter.fillRect(option.rect.adjusted(1, 1, -1, -1), strip)
            return
        if index.data(_OVERVIEW_NOT_LOADED_ROLE):
            painter.save()
            hatch = QColor(opts.palette.color(QPalette.PlaceholderText))
            hatch.setAlpha(90)
            painter.fillRect(option.rect.adjusted(1, 1, -1, -1), QBrush(hatch, Qt.BDiagPattern))
            painter.setPen(opts.palette.color(QPalette.PlaceholderText))
            text = option.fontMetrics.elidedText(str(index.data(Qt.DisplayRole) or "").replace("\n", " · "),
                                                 Qt.ElideRight, option.rect.width() - 4)
            painter.drawText(option.rect, Qt.AlignCenter, text)
            painter.restore()
            return
        option.widget.style().drawPrimitive(QStyle.PE_PanelItemViewItem, opts, painter, option.widget)
        painter.save()
        rect = option.rect.adjusted(1, 1, -1, -1)
        height = option.fontMetrics.height() + 3
        if rect.height() < height + 20:
            height = 0
        image_rect = QRect(rect.left(), rect.top(), rect.width(), max(1, rect.height() - height))
        icon = index.data(Qt.DecorationRole)
        if icon is not None:
            icon.paint(painter, image_rect, Qt.AlignCenter)
        caption = str(index.data(Qt.DisplayRole) or "").split("\n", 1)[0]
        caption = option.fontMetrics.elidedText(caption, Qt.ElideRight, rect.width())
        painter.setPen(opts.palette.color(QPalette.HighlightedText
            if option.state & QStyle.State_Selected else QPalette.Text))
        if height:
            painter.drawText(QRect(rect.left(), image_rect.bottom() + 1, rect.width(), height),
                             Qt.AlignHCenter | Qt.AlignVCenter, caption)
        if option.state & QStyle.State_MouseOver:
            painter.fillRect(option.rect.adjusted(1, 1, -1, -1), QColor(35, 130, 245, 65))
        painter.restore()


class _OverviewImageView(ZoomableImageView):
    """Keep zoom and acquisition-axis wheel navigation explicitly separate."""

    axis_step = Signal(int)
    detail_requested = Signal()

    def __init__(self, parent=None):
        """Create a normal image view with an explicit wheel mode.

        :param parent: the owning browser widget.
        """
        super().__init__(parent)
        self.wheel_mode = "zoom"

    def wheelEvent(self, event):
        """Zoom normally or step the chosen acquisition axis, never both.

        :param event: the actual wheel event.
        """
        if self.wheel_mode != "zoom":
            if event.angleDelta().y():
                self.axis_step.emit(1 if event.angleDelta().y() > 0 else -1)
            event.accept()
            return
        super().wheelEvent(event)
        self.detail_requested.emit()


def _overview_pinch(view, event):
    """Zoom ``view`` alone on a touchpad pinch, about the pointer.

    :param view: the Overview image view whose viewport received ``event``.
    :param event: any viewport event; only a zoom gesture is handled.
    :returns: ``True`` when the pinch zoomed ``view``.
    """
    if (event.type() != QEvent.NativeGesture or view.wheel_mode != "zoom"
            or event.gestureType() != Qt.ZoomNativeGesture or view._item is None):
        return False
    from .cursor_zoom import zoom_at_pointer

    factor = max(.5, min(2., 1. + float(event.value())))
    view._user_zoomed = True
    zoom_at_pointer(view, factor, event.position())
    view.detail_requested.emit()
    event.accept()
    return True


def _overview_set_keeping_view(view, pixmap):
    """Show ``pixmap`` in ``view``, keeping the zoom and centre the user chose.

    A quality change swaps a preview for a larger or smaller one of the same
    image, so the zoom is scaled by the size ratio and the same image point
    stays in the middle of the view.

    :param view: an Overview image view.
    :param pixmap: the new picture of the same image.
    """
    from PySide6.QtCore import QPointF

    previous = view._item
    if previous is None or not view._user_zoomed or previous.pixmap().isNull() or pixmap.isNull():
        view.set_pixmap(pixmap)
        return
    old = previous.pixmap().size()
    center = view.mapToScene(view.viewport().rect().center())
    normalized = QPointF(center.x() / old.width(), center.y() / old.height())
    transform = view.transform()
    view.set_pixmap(pixmap)
    x_ratio, y_ratio = pixmap.width() / old.width(), pixmap.height() / old.height()
    view.setTransform(transform)
    view.scale(1. / x_ratio, 1. / y_ratio)
    view.centerOn(QPointF(normalized.x() * pixmap.width(), normalized.y() * pixmap.height()))
    view._user_zoomed = True


def _fit_overview_action_buttons(buttons):
    """Reserve fresh native text slots without imposing fixed button sizes.

    :param buttons: owned action buttons, measured using their current font,
        style, default-button features and icon insets in actual pixels.
    :returns: ``None``; only minimum dimensions change and may later shrink.
    """
    from PySide6.QtCore import QRect
    from PySide6.QtWidgets import QStyle, QStyleOptionButton
    from ..gui_scale import _set_actual_dim

    for button in buttons:
        button.ensurePolished()
        hint = button.sizeHint().expandedTo(button.minimumSizeHint())
        option = QStyleOptionButton()
        option.initFrom(button)
        option.rect = QRect(0, 0, hint.width(), hint.height())
        option.text, option.icon = button.text(), button.icon()
        option.iconSize = button.iconSize()
        if button.isDefault():
            option.features |= QStyleOptionButton.DefaultButton
        if button.autoDefault():
            option.features |= QStyleOptionButton.AutoDefaultButton
        if button.isFlat():
            option.features |= QStyleOptionButton.Flat
        field = button.style().subElementRect(QStyle.SE_PushButtonContents, option, button)
        text = button.style().itemTextRect(button.fontMetrics(), field,
            int(Qt.AlignCenter | Qt.TextSingleLine), button.isEnabled(), button.text())
        width = max(hint.width(), hint.width() - field.width() + text.width())
        height = max(hint.height(), hint.height() - field.height() + text.height())
        _set_actual_dim(button, "minw", width)
        _set_actual_dim(button, "minh", height)


def _fit_overview_action_width(button, width, context):
    """Fit an owned action to its viewport and restore its selected font on growth.

    Owned actions have empty local sheets; ordinary ``QPushButton`` rules are
    also retained. This is not a general stylesheet selector rewriter.

    :param button: action whose original local stylesheet must be preserved.
    :param width: actual viewport width, or zero for an unexposed action row.
    :param context: font/theme identity; a change re-resolves the base style.
    :returns: ``None``; only this button's local font override and minima change.
    """
    from PySide6.QtGui import QFont, QFontInfo
    from PySide6.QtWidgets import QWidget
    from ..gui_scale import _original_for

    setter = _original_for(button, "setStyleSheet") or QWidget.setStyleSheet
    getter = _original_for(button, "styleSheet") or QWidget.styleSheet
    actual = getter(button)
    state = getattr(button, "_overview_font_fit", None)
    changed = (state is None or state["context"] != context
               or actual != state["applied"] or button.font() != state["font"])
    if not changed and state["width"] == width:
        return
    if changed:
        base = actual if state is None or actual != state["applied"] else state["base"]
        if actual != base:
            setter(button, base)
        button.ensurePolished()
        selected = QFont(button.font())
        state = {"base": base, "selected": selected, "context": context}
    selected_px = QFontInfo(state["selected"]).pixelSize()

    def apply(pixels):
        """Apply a font size and return the refreshed native minimum width."""
        sheet = state["base"] if pixels == selected_px else (
            state["base"] + f"\nQPushButton {{ font-size: {pixels}px; }}")
        if getter(button) != sheet:
            setter(button, sheet)
            if QFontInfo(button.font()).pixelSize() != pixels:
                style = button.style()
                style.unpolish(button)
                style.polish(button)
        button.ensurePolished()
        _fit_overview_action_buttons((button,))
        return button.property("minimumSize").width()

    if width <= 0 or apply(selected_px) <= width:
        apply(selected_px)
    else:
        low, high, best = 1, max(1, selected_px - 1), 1
        while low <= high:
            middle = (low + high) // 2
            if apply(middle) <= width:
                best, low = middle, middle + 1
            else:
                high = middle - 1
        apply(best)
    state.update(applied=getter(button), font=QFont(button.font()), width=width)
    button._overview_font_fit = state


_OVERVIEW_METADATA_TOOLTIP = "Show or hide the filename and metadata text over the images."
_OVERVIEW_QUALITY_STEPS = (128, 256, 512, 1024, 2048, 4096, 0)
"""Image-quality positions: the longest edge a view loads, and 0 at the top for the original pixels."""
_OVERVIEW_QUALITY_DEBOUNCE_MS = 200
_OVERVIEW_QUALITY_TOOLTIP = ("Image quality: the longest edge the plate, well and image views load, from 128 "
                             "pixels up to Original, the full resolution, at the top")
_OVERVIEW_LINK_ZOOM_TOOLTIP = ("Zoom and pan every open image together. Off, only the image under the pointer "
                               "zooms.")
_OVERVIEW_MAX_IMAGE_VIEWS = 6
_OVERVIEW_THUMBNAIL_WORKERS = max(2, min(8, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                                          else (os.cpu_count() or 2)))
_OVERVIEW_FIRST_PAINT_WAIT_S = 0.15
_OVERVIEW_LIBRARY_WAIT_S = 1.5
_OVERVIEW_INSTALL_BUDGET_S = 0.006
_OVERVIEW_THUMBNAIL_BATCH = 8
_OVERVIEW_THUMBNAIL_CACHE = 64
_OVERVIEW_TILE_DEFAULT = 128
_OVERVIEW_COLLAPSED_STRIP = 10
_OVERVIEW_TILE_SPACING = 4
_OVERVIEW_TILE_SPACING_MAX = 64
_OVERVIEW_TILE_RANGE = (48, 768)
_OVERVIEW_MAX_WELL_VIEWS = 6


def _overview_panel_qss(palette, opacity):
    """Give the plate, well and image levels one translucent rounded surface.

    :param palette: the active palette with rendered surface roles.
    :param opacity: the page-opacity preference, passed through unchanged.
    :returns: the QSS block for every overview level container.
    """
    from ..theme import RADIUS, block_surface

    surface = block_surface("surface_alt", palette["theme"], opacity)
    return (f"QFrame#FieldOverviewPanel {{ background: {surface}; "
            f"border: 1px solid {palette['border']}; border-radius: {RADIUS['md']}px; }}\n"
            f"QLabel#FieldOverviewAdjustedBadge {{ background: {palette.get('accent', palette['border'])}; "
            f"color: {palette.get('bg', '#ffffff')}; "
            f"border-radius: {RADIUS['sm']}px; padding: 1px 6px; }}\n"
            f"QLabel#FieldOverviewImageMetadata {{ background: rgba(0, 0, 0, 150); "
            f"color: white; border-radius: {RADIUS['sm']}px; padding: 4px; }}")


def _overview_comparison_frame(source):
    """Read a source's image-level measurements joined with their annotations.

    :param source: the indexed image source folder.
    :returns: one row per measured image.
    :raises _OverviewValidationError: before any measurements were exported.
    """
    from ...measure import _image_annotation_database, _image_measurement_frame

    path = _image_annotation_database(source)
    frame = _image_measurement_frame(path) if path.is_file() else None
    if frame is None or frame.empty:
        raise _OverviewValidationError(tr(
            "Export image-level measurements first; comparisons read them from {path}.", path=path))
    return frame


def _overview_group_label(definition):
    """The label a comparison gives one group, such as ``HeLa + WT``.

    :param definition: ``{annotation column: value}``; omitted columns are pooled.
    :returns: the values joined as :func:`spacr.sp_stats._image_groups` joins them.
    """
    return " + ".join(str(value) for value in dict(definition).values())


def _overview_all_combinations(levels):
    """Every combination of the chosen values of the chosen columns.

    :param levels: ``{annotation column: [values]}``.
    :returns: one ``{column: value}`` group per combination, in column order.
    """
    from itertools import product

    columns = [column for column, values in levels.items() if values]
    return [dict(zip(columns, values)) for values in product(*(levels[column] for column in columns))]


def _overview_vary_one(column, values, held):
    """Groups that differ only in ``column``, every other column held fixed.

    :param column: the annotation column that varies.
    :param values: its values, one group each.
    :param held: ``{other column: value}`` kept the same in every group.
    :returns: the groups, ``held`` first and ``column`` last in each.
    """
    held = {key: value for key, value in dict(held).items() if key != column}
    return [dict(held, **{column: value}) for value in values]


def _overview_grouped_frame(frame, groups, names=None):
    """Label each image with the built group it belongs to, in group order.

    An image matching several groups appears once per group, so each group
    gets its own representative.

    :param frame: one row per image with annotation columns.
    :param groups: ``{column: value}`` groups; omitted columns are pooled.
    :param names: the groups' names, ``""`` for an unnamed one.
    :returns: the matching rows with an ``_overview_group`` label column.
    """
    import pandas as pd

    names = list(names or ())
    parts = []
    for index, definition in enumerate(groups):
        mask = pd.Series(True, index=frame.index)
        for key, value in definition.items():
            if key not in frame.columns:
                raise KeyError(key)
            mask &= frame[key].astype(str) == str(value)
        name = names[index] if index < len(names) else ""
        parts.append(frame[mask].assign(_overview_group=name or _overview_group_label(definition)))
    return pd.concat(parts) if parts else frame.iloc[0:0].assign(_overview_group="")


def _overview_group_count(fields, annotations, definition):
    """How many fields lie in wells whose annotations match ``definition``.

    :param fields: the indexed fields.
    :param annotations: ``{(plate, well): {column: value}}``.
    :param definition: ``{column: value}``; omitted columns match anything.
    :returns: the number of matching fields.
    """
    def matches(field):
        """Whether the field's well annotations hold every value of the group."""
        metadata = {"plate": field.plate, **annotations.get((field.plate, field.well), {})}
        return all(str(metadata.get(key)) == str(value) for key, value in definition.items())
    return sum(1 for field in fields if matches(field))


_OVERVIEW_GROUP_NAME_ROLE = Qt.UserRole + 2
_OVERVIEW_NOT_LOADED_ROLE = Qt.UserRole + 4


class _OverviewGroupBuilder(QWidget):
    """Build comparison groups from values, or "any", of each annotation column.

    Its three parts -- the per-column value pickers, the group list with its
    actions, and the presets -- are laid out as separate setting rows.
    """

    changed = Signal()

    def __init__(self, counter, levels, parent=None):
        """Create empty pickers and an empty group list.

        :param counter: ``counter(definition) -> int`` images in a group.
        :param levels: ``levels() -> {column: [values]}`` ticked for comparison.
        :param parent: the owning browser.
        """
        from PySide6.QtWidgets import QFormLayout, QListWidget
        from .setup_dialog import _SetupComboBox

        super().__init__(parent)
        self._counter, self._levels = counter, levels
        self._pickers = {}
        self.picker_host = QWidget(self)
        self._picker_form = QFormLayout(self.picker_host)
        self._picker_form.setContentsMargins(0, 0, 0, 0)
        self._empty = QLabel(tr("Annotate wells to build groups."))
        self._empty.setWordWrap(True)
        self._picker_form.addRow(self._empty)
        self.groups = QListWidget()
        self.groups.setObjectName("OverviewComparisonGroups")
        self.groups.setToolTip(tr("Groups compared, in order; each lists its label and image count."))
        self.groups.setMinimumHeight(96)
        self.groups.itemDoubleClicked.connect(self._rename_clicked)
        self.group_host = QWidget(self)
        group_layout = QVBoxLayout(self.group_host)
        group_layout.setContentsMargins(0, 0, 0, 0)
        group_layout.addWidget(self.groups)
        actions = QHBoxLayout()
        self.buttons = {}
        for name, caption, tip, callback in (
                ("add", "Add group", "Add a group with the values chosen above; columns left at Any are pooled.", self.add_group),
                ("remove", "Remove group", "Remove the selected group.", self.remove_group),
                ("up", "Move up", "Move the selected group earlier.", lambda: self.move_group(-1)),
                ("down", "Move down", "Move the selected group later.", lambda: self.move_group(1)),
                ("name", "Name group", "Give the selected group a name; its images and representative are shown under that name.", self.rename_group),
                ("clear", "Clear groups", "Remove every group; comparisons then use the ticked values.", self.clear_groups)):
            button = QPushButton(tr(caption))
            button.setToolTip(tr(tip))
            button.clicked.connect(lambda _checked=False, run=callback: run())
            actions.addWidget(button)
            self.buttons[name] = button
        group_layout.addLayout(actions)
        self.preset_host = QWidget(self)
        presets = QHBoxLayout(self.preset_host)
        presets.setContentsMargins(0, 0, 0, 0)
        self.preset = _SetupComboBox()
        self.preset.addItem(tr("All combinations of the ticked values"), "combinations")
        self.preset.addItem(tr("Vary one column, hold the others"), "vary")
        self.preset.setToolTip(tr(
            "All combinations makes one group per combination of the values ticked under Annotation "
            "dimensions; Vary one column makes one group per ticked value of the varied column and holds "
            "the other columns at the values chosen above."))
        self.vary = _SetupComboBox()
        self.vary.setToolTip(tr("Annotation column that varies between the preset's groups."))
        apply = QPushButton(tr("Add preset groups"))
        apply.setToolTip(tr("Replace the groups with the preset's groups."))
        apply.clicked.connect(self.apply_preset)
        self.buttons["preset"] = apply
        for widget in (self.preset, self.vary, apply):
            presets.addWidget(widget)
        self.hide()

    def set_columns(self, values):
        """Offer every annotation column and value, keeping earlier choices.

        :param values: ``{column: set of values}`` from the annotations.
        """
        from .setup_dialog import _SetupComboBox

        chosen = self.definition()
        while self._picker_form.rowCount() > 1:
            self._picker_form.removeRow(1)
        self._pickers.clear()
        self._empty.setVisible(not values)
        old_vary = self.vary.currentData()
        self.vary.clear()
        for column in sorted(values):
            combo = _SetupComboBox()
            combo.addItem(tr("Any"), None)
            for value in sorted(values[column], key=_overview_natural_key):
                combo.addItem(str(value), value)
            combo.setCurrentIndex(max(0, combo.findData(chosen.get(column))))
            combo.setToolTip(tr("Value of {column} in the next group; Any pools every value.", column=column))
            label = QLabel(column)
            label.setToolTip(combo.toolTip())
            self._picker_form.addRow(label, combo)
            self._pickers[column] = combo
            self.vary.addItem(column, column)
        self.vary.setCurrentIndex(max(0, self.vary.findData(old_vary)))
        self.refresh_counts()

    def definition(self):
        """The group the pickers describe: ``{column: value}`` without Any columns."""
        return {column: combo.currentData() for column, combo in self._pickers.items()
                if combo.currentData() is not None}

    def definitions(self):
        """Every group in order, as ``{column: value}`` mappings."""
        return [dict(self.groups.item(index).data(Qt.UserRole)) for index in range(self.groups.count())]

    def names(self):
        """Every group's name in order; an unnamed group gives ``""``."""
        return [str(self.groups.item(index).data(_OVERVIEW_GROUP_NAME_ROLE) or "")
                for index in range(self.groups.count())]

    def _caption(self, definition, name=""):
        """A group's name, label and image count."""
        label = _overview_group_label(definition) or tr("Any")
        if name:
            return tr("{name} ({label}) — {count} images", name=name, label=label,
                      count=self._counter(definition))
        return tr("{label} — {count} images", label=label, count=self._counter(definition))

    def rename_group(self, *_, row=None, name=None):
        """Name a group; an empty name returns it to its values.

        :param row: the group; the selected one when omitted.
        :param name: the name; asked for when omitted.
        :returns: whether the group now has the name.
        """
        from PySide6.QtWidgets import QInputDialog

        row = self.groups.currentRow() if row is None else row
        item = self.groups.item(row) if row is not None and row >= 0 else None
        if item is None:
            return False
        if name is None:
            name, accepted = QInputDialog.getText(self, tr("Name group"), tr("Group name"),
                                                  text=str(item.data(_OVERVIEW_GROUP_NAME_ROLE) or ""))
            if not accepted:
                return False
        name = str(name).strip()
        taken = {other for index, other in enumerate(self.names()) if index != row and other}
        taken |= {_overview_group_label(group) for index, group in enumerate(self.definitions()) if index != row}
        if name and name in taken:
            return False
        item.setData(_OVERVIEW_GROUP_NAME_ROLE, name)
        item.setText(self._caption(item.data(Qt.UserRole), name))
        self.changed.emit()
        return True

    def add_group(self, definition=None):
        """Append a group, refusing an empty or repeated one.

        :param definition: the group; the pickers' group when omitted.
        :returns: whether a group was added.
        """
        from PySide6.QtWidgets import QListWidgetItem

        definition = dict(self.definition() if definition is None else definition)
        labels = {_overview_group_label(group) for group in self.definitions()}
        if not definition or _overview_group_label(definition) in labels:
            return False
        item = QListWidgetItem(self._caption(definition))
        item.setData(Qt.UserRole, definition)
        self.groups.addItem(item)
        self.changed.emit()
        return True

    def _rename_clicked(self, item):
        """Name the double-clicked group."""
        self.rename_group(row=self.groups.row(item))

    def remove_group(self):
        """Remove the selected group."""
        row = self.groups.currentRow()
        if row >= 0:
            self.groups.takeItem(row)
            self.changed.emit()

    def move_group(self, delta):
        """Move the selected group earlier or later.

        :param delta: -1 for earlier, 1 for later.
        """
        row = self.groups.currentRow()
        target = row + delta
        if row < 0 or not 0 <= target < self.groups.count():
            return
        item = self.groups.takeItem(row)
        self.groups.insertItem(target, item)
        self.groups.setCurrentRow(target)
        self.changed.emit()

    def clear_groups(self):
        """Remove every group."""
        self.groups.clear()
        self.changed.emit()

    def apply_preset(self):
        """Replace the groups with the chosen preset's groups.

        :returns: the number of groups added.
        """
        levels = self._levels()
        if self.preset.currentData() == "vary":
            column = self.vary.currentData()
            combo = self._pickers.get(column)
            if combo is None:
                return 0
            values = levels.get(column) or [combo.itemData(index) for index in range(1, combo.count())]
            groups = _overview_vary_one(column, values, self.definition())
        else:
            groups = _overview_all_combinations(levels)
        self.groups.clear()
        return sum(self.add_group(group) for group in groups)

    def refresh_counts(self):
        """Recount every group's images after annotations change."""
        for index in range(self.groups.count()):
            item = self.groups.item(index)
            item.setText(self._caption(item.data(Qt.UserRole),
                                       str(item.data(_OVERVIEW_GROUP_NAME_ROLE) or "")))


class _OverviewCollapseArrow(QToolButton):
    """Paint spaCR's standard container-collapse tab on a level header."""

    def __init__(self, parent=None):
        """Create a checkable arrow whose checked state means collapsed.

        :param parent: the owning level container.
        """
        super().__init__(parent)
        self.setCheckable(True)
        self.setAutoRaise(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setToolTip(tr("Collapse or expand this container"))
        self.setAccessibleName(tr("Collapse or expand this container"))

    def sizeHint(self):
        """Keep the shared handle proportions at the current font size.

        :returns: a horizontal tab three text lines wide.
        """
        from PySide6.QtCore import QSize

        height = max(12, self.fontMetrics().height())
        return QSize(3 * height, height)

    def minimumSizeHint(self):
        """Never squeeze the arrow below its painted tab.

        :returns: the same size as :meth:`sizeHint`.
        """
        return self.sizeHint()

    def paintEvent(self, _event):
        """Point up while open and down while collapsed.

        :param _event: the paint event; the whole handle is drawn.
        """
        from PySide6.QtGui import QPainter
        from ..theme import active_palette
        from .collapse_arrow import paint_collapse_arrow

        painter = QPainter(self)
        paint_collapse_arrow(painter, self.rect(), Qt.Vertical, not self.isChecked(),
                             active_palette() or {}, self.underMouse() or self.hasFocus())
        painter.end()


class _OverviewPanel(QFrame):
    """One translucent, rounded and collapsible plate, well or image level."""

    collapsed_changed = Signal(bool)

    def __init__(self, title, parent=None, *, shrink_width=False):
        """Create a header with the shared collapse arrow above a body.

        :param title: the translated level caption.
        :param parent: the owning overview.
        :param shrink_width: also release horizontal space while collapsed.
        """
        super().__init__(parent)
        self.setObjectName("FieldOverviewPanel")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self._shrink_width = shrink_width
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 4, 8, 8)
        outer.setSpacing(4)
        self.header_layout = QHBoxLayout()
        self.header_layout.setContentsMargins(0, 0, 0, 0)
        self.arrow = _OverviewCollapseArrow(self)
        self.arrow.toggled.connect(self._apply_collapsed)
        self.title = QLabel(title)
        self.header_layout.addWidget(self.arrow)
        self.header_layout.addWidget(self.title, 1)
        outer.addLayout(self.header_layout)
        self.body = QWidget(self)
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.body, 1)

    def set_collapsed(self, collapsed):
        """Minimise the level to its header or restore its body.

        :param collapsed: whether only the header remains.
        """
        self.arrow.setChecked(bool(collapsed))

    def is_collapsed(self):
        """Report whether the level shows only its header.

        :returns: the arrow's checked state.
        """
        return self.arrow.isChecked()

    def _apply_collapsed(self, collapsed):
        """Release the body's space to the neighbouring levels.

        :param collapsed: the arrow's new checked state.
        """
        from ..gui_scale import _set_actual_dim

        self.body.setVisible(not collapsed)
        hint = self.sizeHint()
        _set_actual_dim(self, "maxh", hint.height() if collapsed else 16777215)
        if self._shrink_width:
            _set_actual_dim(self, "maxw", hint.width() if collapsed else 16777215)
        self.collapsed_changed.emit(collapsed)


def _overview_store_groups(source, groups, names):
    """Save comparison groups and names for ``source``, off the GUI thread.

    :param source: the image source folder.
    :param groups: ``{column: value}`` definitions.
    :param names: one name per group, ``""`` for unnamed.
    :returns: the stored names.
    """
    from ...measure import _image_annotation_database, _save_comparison_groups

    return _save_comparison_groups(_image_annotation_database(source), groups, names)


def _overview_saved_groups(source):
    """Read the comparison groups saved for ``source``, off the GUI thread.

    :mod:`spacr.measure` is imported only when a measurements database exists:
    a source without one has no saved groups, and the import costs several
    seconds of CPU that a fresh Overview should not spend on every open.

    :param source: the image source folder.
    :returns: ``(groups, names)``.
    """
    candidate = Path(source).expanduser()
    database = candidate if candidate.suffix == ".db" else candidate / "measurements" / "measurements.db"
    if not database.is_file():
        return [], []
    from ...measure import _read_comparison_groups

    return _read_comparison_groups(database)


def _overview_import_scheme(source, path, records):
    """Write a well annotation scheme into the source's measurements database.

    :param source: the image source folder.
    :param path: the scheme file.
    :param records: the source's fields as mappings.
    :returns: :func:`spacr.measure._import_well_annotation_scheme`'s summary.
    """
    from ...measure import _image_annotation_database, _import_well_annotation_scheme

    return _import_well_annotation_scheme(_image_annotation_database(source), path, records, source=source)


def _overview_scheme_preview(path):
    """Read a well annotation scheme for its preview, off the GUI thread.

    :param path: the scheme file.
    :returns: the frame :func:`spacr.measure._read_well_annotation_scheme` returns.
    """
    from ...measure import _read_well_annotation_scheme

    return _read_well_annotation_scheme(path)


def _overview_scheme_annotations(frame, fields):
    """The ``{(plate, well): {column: value}}`` a scheme gives this source's wells.

    :param frame: the scheme, one row per well, from the preview.
    :param fields: the indexed fields.
    :returns: ``(annotations, unmatched)``: the annotations keyed by the
        source's own well names, and how many scheme rows match no well.
    """
    from ...measure import _image_annotation_column_name, _scheme_well

    present = {}
    for field in fields:
        if not field.unresolved:
            present.setdefault((str(field.plate), _scheme_well(field.well)), (field.plate, field.well))
    plates = sorted({plate for plate, _well in present})
    columns = {column: _image_annotation_column_name(column) for column in list(frame.columns)[2:]}
    result, unmatched = {}, 0
    for entry in frame.to_dict("records"):
        addresses = [present[(str(plate), entry["well"])] for plate in ([entry["plate"]] if entry["plate"] else plates)
                     if (str(plate), entry["well"]) in present]
        unmatched += not addresses
        for address in addresses:
            for original, column in columns.items():
                value = entry.get(original)
                if value is not None and str(value) != "nan":
                    result.setdefault(address, {})[column] = str(value)
    return result, unmatched


class _OverviewAnnotationDialog(QDialog):
    """Ask for one metadata key and value for the selected wells."""

    def __init__(self, keys, wells, parent=None):
        """Offer standard and existing metadata keys with a free value.

        :param keys: existing annotation keys in display order.
        :param wells: the plate-qualified wells that will be annotated.
        :param parent: the owning overview.
        """
        from PySide6.QtWidgets import QDialogButtonBox, QFormLayout, QLineEdit
        from .setup_dialog import _SetupComboBox

        super().__init__(parent)
        self.setWindowTitle(tr("Annotate selected wells"))
        self.wells = tuple(wells)
        form = QFormLayout(self)
        form.addRow(QLabel(tr("One selected well") if len(self.wells) == 1
                           else tr("{count} selected wells", count=len(self.wells))))
        self.key = _SetupComboBox()
        self.key.setEditable(True)
        for label, key in ((tr("Condition"), "condition"), (tr("Cell type"), "cell_type"),
                               (tr("Pathogen"), "pathogen_type"), (tr("Genotype"), "genotype"),
                           (tr("Drug"), "drug"), (tr("Replicate"), "replicate")):
            self.key.addItem(label, key)
        for key in keys:
            if self.key.findData(key) < 0:
                self.key.addItem(key, key)
        self.value = QLineEdit()
        self.value.setPlaceholderText(tr("Value"))
        form.addRow(tr("Metadata"), self.key)
        form.addRow(tr("Value"), self.value)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def annotation(self):
        """Return the chosen key and stripped value.

        A listed choice keeps its stable key; typed text is used literally.

        :returns: ``(key, value)``.
        """
        index = self.key.currentIndex()
        if index >= 0 and self.key.currentText() == self.key.itemText(index):
            return self.key.currentData(), self.value.text().strip()
        return self.key.currentText().strip(), self.value.text().strip()


class _OverviewLevelsHistogram(QWidget):
    """A channel's display histogram with draggable black, white and gamma points.

    The horizontal axis is the display level, 0 to 1, after the global
    normalization; the black and white handles map onto it directly and the
    gamma handle sits at the level shown as mid-grey.
    """

    levels_changed = Signal(float, float, float)
    released = Signal()

    def __init__(self, parent=None):
        """Create an empty histogram with the identity levels.

        :param parent: the image settings panel.
        """
        super().__init__(parent)
        self.setObjectName("OverviewLevelsHistogram")
        self.setMinimumSize(220, 96)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)
        self.setProperty("spacrOwnsMouseGesture", True)
        self.setCursor(Qt.SizeHorCursor)
        self.counts = None
        self.black, self.white, self.gamma = 0., 1., 1.
        self._dragging = None

    def set_counts(self, counts):
        """Draw ``counts`` (256 bins over 0-1), or nothing for ``None``."""
        self.counts = None if counts is None else np.asarray(counts, dtype=np.float64)
        self.update()

    def set_levels(self, black, white, gamma):
        """Move the handles without emitting a change."""
        self.black, self.white, self.gamma = float(black), float(white), float(gamma)
        self.update()

    def _plot(self):
        """The rectangle the histogram fills, inside the handle margins."""
        return QRect(8, 4, max(1, self.width() - 16), max(1, self.height() - 16))

    def _x(self, level):
        """The widget x of a 0-1 level."""
        plot = self._plot()
        return plot.left() + level * plot.width()

    def _level(self, x):
        """The 0-1 level at widget x."""
        plot = self._plot()
        return min(max((x - plot.left()) / plot.width(), 0.), 1.)

    def _handles(self):
        """The handles as ``(name, level)``, black, gamma and white."""
        return (("black", self.black), ("gamma", _overview_gamma_midpoint(self.black, self.white, self.gamma)),
                ("white", self.white))

    def paintEvent(self, _event):
        """Draw the bars, the shaded clipped ranges and the three handles."""
        from PySide6.QtGui import QPainter, QPen, QPolygonF
        from PySide6.QtCore import QPointF, QRectF
        from ..theme import _painting_palette

        palette = _painting_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        plot = self._plot()
        painter.fillRect(plot, QColor(palette["surface_alt"]))
        painter.setPen(QPen(QColor(palette["border"]), 1))
        painter.drawRect(plot)
        if self.counts is not None and self.counts.size and self.counts.max() > 0:
            heights = np.sqrt(self.counts / self.counts.max())
            bar = QColor(palette["accent"])
            bar.setAlpha(170)
            width = plot.width() / heights.size
            for index, height in enumerate(heights):
                if height <= 0:
                    continue
                top = plot.bottom() - height * plot.height()
                painter.fillRect(QRectF(plot.left() + index * width, top, max(width, 1.), plot.bottom() - top), bar)
        shade = QColor(palette["fg"])
        shade.setAlpha(40)
        painter.fillRect(QRectF(plot.left(), plot.top(), self._x(self.black) - plot.left(), plot.height()), shade)
        painter.fillRect(QRectF(self._x(self.white), plot.top(), plot.right() - self._x(self.white), plot.height()), shade)
        fills = {"black": QColor("#000000"), "white": QColor("#ffffff"), "gamma": QColor(palette["accent"])}
        for name, level in self._handles():
            x = self._x(level)
            painter.setPen(QPen(QColor(palette["fg"] if name != "gamma" else palette["accent"]), 1))
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            painter.setBrush(fills[name])
            painter.setPen(QPen(QColor(palette["fg"]), 1))
            painter.drawPolygon(QPolygonF([QPointF(x, plot.bottom() + 1), QPointF(x - 6, self.height() - 2),
                                           QPointF(x + 6, self.height() - 2)]))
        painter.end()

    def mousePressEvent(self, event):
        """Pick the nearest handle within reach."""
        x = event.position().x()
        name, level = min(self._handles(), key=lambda handle: abs(self._x(handle[1]) - x))
        self._dragging = name if abs(self._x(level) - x) <= 10 else None
        event.accept()

    def mouseMoveEvent(self, event):
        """Drag the picked handle, keeping black below the midpoint below white."""
        if self._dragging is None:
            return
        level = self._level(event.position().x())
        if self._dragging == "black":
            self.black = min(level, self.white - .01)
        elif self._dragging == "white":
            self.white = max(level, self.black + .01)
        else:
            self.gamma = _overview_gamma_from_midpoint(self.black, self.white, level)
        self.update()
        self.levels_changed.emit(self.black, self.white, self.gamma)

    def mouseReleaseEvent(self, event):
        """End a drag."""
        if self._dragging is not None:
            self._dragging = None
            self.released.emit()
        event.accept()


class _OverviewTileDelegate(QStyledItemDelegate):
    """Draw a well view's tiles and mark the ones with their own display settings."""

    def __init__(self, browser, parent=None):
        """Keep the browser that knows which images are adjusted.

        :param browser: the plate browser.
        :param parent: the well view.
        """
        super().__init__(parent)
        self._browser = browser

    def paint(self, painter, option, index):
        """Paint the tile, then a small "Adjusted" badge on an adjusted image."""
        super().paint(painter, option, index)
        field = index.data(Qt.UserRole)
        if field is None or not self._browser._is_adjusted(field):
            return
        painter.save()
        text = tr("Adjusted")
        metrics = option.fontMetrics
        width, height = metrics.horizontalAdvance(text) + 10, metrics.height() + 2
        rect = QRect(option.rect.right() - width - 4, option.rect.top() + 4, width, height)
        painter.setRenderHint(painter.RenderHint.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(option.palette.color(QPalette.Highlight))
        painter.drawRoundedRect(rect, height / 2, height / 2)
        painter.setPen(option.palette.color(QPalette.HighlightedText))
        painter.drawText(rect, Qt.AlignCenter, text)
        painter.restore()


_OVERVIEW_SETTINGS_ROWS = (
    ("Black point", "overview_image_black_point", "Display level shown as black for this channel of this image; lower levels are clipped. Drag the black handle under the histogram or type a level from 0 to 1."),
    ("White point", "overview_image_white_point", "Display level shown at full brightness for this channel of this image; higher levels are clipped. Drag the white handle or type a level from 0 to 1."),
    ("Gamma", "overview_image_gamma", "Gamma between the black and white points: above 1 brightens the mid-tones, below 1 darkens them. The middle handle is the level shown as mid-grey."),
    ("Brightness", "overview_image_brightness", "Adds to this channel's display level after the levels, from -100 to 100 percent of the display range."),
    ("Contrast", "overview_image_contrast", "Stretches this channel about mid-grey after the levels: 100 percent leaves it unchanged, lower flattens, higher steepens."),
    ("Visible", "overview_image_visible", "Show this channel in this image; switched off, the channel is left out of the composite."),
    ("Saturation", "overview_image_saturation", "Colour saturation of this image's combined colour image, all channels together (it is not set per channel): 0 percent is grey, 100 percent unchanged, higher more vivid."),
    ("Link all channels", "overview_image_link_channels", "Off: an edit changes only the selected channel. On: the same edit is made to every channel of this image."),
    ("Apply to", "overview_image_apply_to", "Copy this image's settings, for the selected channel or for all channels, to the other images of its well, its plate or every loaded image."),
)


class _OverviewImageSettings(QDialog):
    """Right-click settings for one image: per-channel levels, brightness, contrast and visibility.

    Every change is applied at once to that image only, as its entry in the
    browser's per-image overrides, and saved in ``results/overview_view.json``.
    """

    def __init__(self, browser, field, channels):
        """Build the panel for ``field`` and read its histograms in the background.

        :param browser: the plate browser that owns the overrides.
        :param field: the image the settings belong to.
        :param channels: the shown channels, in order.
        """
        from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout, QSlider, QTabBar
        from .setup_dialog import _SetupComboBox

        super().__init__(browser, Qt.Window)
        self.setObjectName("OverviewImageSettings")
        self.setProperty("spacrSettingsPopup", True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setWindowTitle(tr("Image settings"))
        self._browser = browser
        self._loading = False
        self._pending = None
        self._histograms = {}
        self._push_timer = QTimer(self)
        self._push_timer.setSingleShot(True)
        self._push_timer.setInterval(40)
        self._push_timer.timeout.connect(self._push)
        layout = QVBoxLayout(self)
        self.caption = QLabel()
        self.caption.setObjectName("OverviewImageSettingsCaption")
        self.caption.setWordWrap(True)
        self.caption.setCursor(Qt.SizeAllCursor)
        self.caption.setToolTip(tr("Drag this title to move the window."))
        layout.addWidget(self.caption)
        body = QWidget(self)
        body.setObjectName("OverviewImageSettingsBody")
        body.setProperty("spacrOwnsMouseGesture", True)
        outer, layout = layout, QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(body, 1)
        self.tabs = QTabBar()
        self.tabs.setObjectName("OverviewImageSettingsChannels")
        self.tabs.setExpanding(False)
        self.tabs.currentChanged.connect(self._channel_changed)
        layout.addWidget(self.tabs)
        self.histogram = _OverviewLevelsHistogram(body)
        self.histogram.levels_changed.connect(self._levels_dragged)
        self.histogram.released.connect(self._push)
        layout.addWidget(self.histogram)
        self.black, self.white, self.gamma = QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox()
        for spin, low, high, step, decimals in ((self.black, 0., 1., .01, 3), (self.white, 0., 1., .01, 3),
                                                (self.gamma, .05, 10., .05, 2)):
            spin.setRange(low, high)
            spin.setSingleStep(step)
            spin.setDecimals(decimals)
        self.brightness, self.contrast, self.saturation = (QSlider(Qt.Horizontal), QSlider(Qt.Horizontal),
                                                           QSlider(Qt.Horizontal))
        for slider, low, high in ((self.brightness, -100, 100), (self.contrast, 0, 300), (self.saturation, 0, 300)):
            slider.setRange(low, high)
            slider.setPageStep(10)
        self.visible = Toggle()
        self.link = Toggle()
        self.scope = _SetupComboBox()
        for text, value in ((tr("This well"), "well"), (tr("This plate"), "plate"), (tr("All images"), "all")):
            self.scope.addItem(text, value)
        self.apply_channels = _SetupComboBox()
        self.apply_channels.addItem(tr("Selected channel"), "selected")
        self.apply_channels.addItem(tr("All channels"), "all")
        self.apply_button = QPushButton(tr("Apply"))
        self.apply_button.clicked.connect(self._apply_to)
        apply_row = QWidget()
        apply_layout = QHBoxLayout(apply_row)
        apply_layout.setContentsMargins(0, 0, 0, 0)
        for widget in (self.scope, self.apply_channels, self.apply_button):
            apply_layout.addWidget(widget)
        form = QFormLayout()
        keys = {}
        for (caption, key, tooltip), field_widget in zip(_OVERVIEW_SETTINGS_ROWS, (
                self.black, self.white, self.gamma, self.brightness, self.contrast,
                self.visible, self.saturation, self.link, apply_row)):
            field_widget.setProperty("settingKey", key)
            field_widget.setProperty("apiTooltipDescriptionSource", tooltip)
            form.addRow(_overview_row_label(caption, tooltip), field_widget)
            keys[field_widget] = key
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self.reset_button = QPushButton(tr("Reset"))
        self.reset_button.setToolTip(tr("Return every channel of this image to the plate's display settings."))
        self.reset_button.clicked.connect(self._reset)
        close = QPushButton(tr("Close"))
        close.clicked.connect(self.close)
        buttons.addWidget(self.reset_button)
        buttons.addStretch(1)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        for spin, name in ((self.black, "black"), (self.white, "white"), (self.gamma, "gamma")):
            spin.valueChanged.connect(functools.partial(self._edited, name))
        self.brightness.valueChanged.connect(functools.partial(self._scaled_edit, "brightness"))
        self.contrast.valueChanged.connect(functools.partial(self._scaled_edit, "contrast"))
        self.visible.toggled.connect(functools.partial(self._edited, "visible"))
        self.saturation.valueChanged.connect(self._saturation_edited)
        from ..screens.settings_model import install_api_tooltips

        install_api_tooltips(self, "overview", keys)
        self.show_field(field, channels)

    def show_field(self, field, channels):
        """Edit ``field`` from now on, starting on its first shown channel.

        :param field: the image.
        :param channels: the shown channels, in order.
        """
        self._push()
        self.field, self.key = field, _overview_image_key(field)
        self.channels = tuple(channels) or (0,)
        names = self._browser._overview_options.get("channel_names") or {}
        self.caption.setText(tr("Plate {plate}, well {well}, field {field}", plate=field.plate,
                                well=field.well or tr("Unresolved"), field=field.field))
        self._loading = True
        try:
            while self.tabs.count():
                self.tabs.removeTab(0)
            for channel in self.channels:
                self.tabs.addTab(names.get(channel, names.get(str(channel), "")) or
                                 tr("Channel {channel}", channel=channel))
            self.tabs.setCurrentIndex(0)
        finally:
            self._loading = False
        self._histograms = {}
        self._load_values()
        self._browser._request_image_histograms(self)

    @property
    def channel(self):
        """The channel being edited."""
        return self.channels[max(0, min(self.tabs.currentIndex(), len(self.channels) - 1))]

    def _override(self):
        """This image's override as edited so far."""
        return self._pending if self._pending is not None else self._browser._image_overrides.get(self.key)

    def _load_values(self):
        """Show the selected channel's settings without writing them back."""
        override = self._override()
        settings = _overview_channel_override(override, self.channel)
        self._loading = True
        try:
            self.black.setValue(settings["black"])
            self.white.setValue(settings["white"])
            self.gamma.setValue(settings["gamma"])
            self.brightness.setValue(round(settings["brightness"] * 100))
            self.contrast.setValue(round(settings["contrast"] * 100))
            self.visible.setChecked(bool(settings["visible"]))
            self.saturation.setValue(round(float((override or {}).get("saturation", 1.)) * 100))
            self.histogram.set_levels(settings["black"], settings["white"], settings["gamma"])
            self.histogram.set_counts(self._histograms.get(self.channel))
        finally:
            self._loading = False

    def _channel_changed(self, *_):
        """Show the newly selected channel's settings and histogram."""
        if not self._loading:
            self._load_values()

    def set_histograms(self, histograms):
        """Keep the read histograms and draw the selected channel's.

        :param histograms: ``{channel: counts}``, or the reason they are missing.
        """
        self._histograms = histograms if isinstance(histograms, dict) else {}
        self.histogram.set_counts(self._histograms.get(self.channel))
        self.histogram.setToolTip("" if isinstance(histograms, dict) else str(histograms))

    def _scaled_edit(self, name, value):
        """A percentage slider moved: edit its setting in display units."""
        self._edited(name, value / 100.)

    def _levels_dragged(self, black, white, gamma):
        """A histogram handle moved: edit the three levels together."""
        self._edit({"black": black, "white": white, "gamma": gamma})

    def _edited(self, name, value):
        """One control changed: edit that setting."""
        self._edit({name: bool(value) if name == "visible" else float(value)})

    def _edit(self, values):
        """Write ``values`` into the selected channel, or every channel when linked, and show them.

        :param values: ``{setting: value}``.
        """
        import copy

        if self._loading:
            return
        override = copy.deepcopy(self._override() or {})
        channels = override.setdefault("channels", {})
        for channel in (self.channels if self.link.isChecked() else (self.channel,)):
            settings = _overview_channel_override(override, channel)
            settings.update(values)
            if settings["white"] <= settings["black"]:
                if "black" in values and "white" not in values:
                    settings["black"] = max(0., settings["white"] - .01)
                else:
                    settings["white"] = min(1., settings["black"] + .01)
            channels.pop(channel, None)
            channels[str(channel)] = settings
        self._pending = override
        self._load_values()
        self._push_timer.start()

    def _saturation_edited(self, value):
        """The saturation slider moved: edit the composite's saturation."""
        import copy

        if self._loading:
            return
        override = copy.deepcopy(self._override() or {})
        override["saturation"] = value / 100.
        self._pending = override
        self._push_timer.start()

    def _push(self):
        """Hand the edited override to the browser, which redraws and saves it."""
        self._push_timer.stop()
        if self._pending is None:
            return
        pending, self._pending = self._pending, None
        self._browser._set_image_override(self.key, pending)

    def _reset(self):
        """Return every channel of this image to the plate's display settings."""
        self._push_timer.stop()
        self._pending = None
        self._browser._set_image_override(self.key, None)
        self._load_values()

    def _apply_to(self):
        """Copy this image's settings to its well, its plate or every image."""
        self._push()
        channel = self.channel if self.apply_channels.currentData() == "selected" else None
        self._browser._copy_image_override(self.field, self.scope.currentData(), channel)

    def closeEvent(self, event):
        """Save a last unsent change before closing."""
        self._push()
        super().closeEvent(event)


class _FieldOverview(QWidget):
    """Shared, lazily loaded plate and field overview for both mask modules."""

    measurement_progress = Signal(int, int)
    _channels_found = Signal(object)

    def __init__(self, parent=None, settings=None):
        """Create controls without opening a directory or decoding an image.

        :param parent: the module that owns this retained browser.
        :param settings: accepted metadata, geometry and display configuration.
        """
        from collections import OrderedDict
        from PySide6.QtWidgets import (QLineEdit, QSpinBox, QDoubleSpinBox,
                                       QAbstractItemView, QTableWidget)
        import threading

        super().__init__(parent)
        from ..gil_priority import hold_background_imports

        hold_background_imports(self, _FieldOverview._loading)
        self._overview_options = dict(settings or {})
        self._advanced_model = None
        self._advanced_enabled = False
        self._channel_overrides = {}
        self._loading_controls = True
        self._selected_wells = set()
        self._selected_well_axes = {}
        self._selected_well_options = {}
        self._selected_field_options = {}
        self._source_shapes = {}
        self._native_item = self._native_key = None
        self._native_pending = False
        self._metadata_source = ""
        self._source_paths = []
        self._all_fields = ()
        self._load_limits = (None, None)
        self._annotations = {}
        self._metadata_revision = 0
        self._annotation_write_lock = threading.Lock()
        self._image_annotation_queue = []
        self._image_annotation_running = False
        self._annotation_file_actions = 0
        self._annotation_write_running = False
        self._annotation_pending_snapshot = {}
        self._measurement_cancelled = threading.Event()
        self._drag_start, self._drag_selected = None, False
        self._normalization_plan = None
        self._normalization_context = None
        self._normalization_plate = None
        self._normalization_ready = not bool(self._overview_options.get("normalization"))
        self._display_pending = False
        self._display_generation = 0
        self._conditions = dict(self._overview_options.get("conditions")
                                or (self._overview_options.get("normalization") or {}).get("conditions") or {})
        self._annotations = {address: {"condition": value} for address, value in self._conditions.items()}
        self._preview_edge = 256
        self._enhancement = None
        self._enhance_channels = None
        self._normalize_channels = None
        self._original_image = False
        self._link_zoom = False
        self._quality_sliders = []
        self._tile_size = _OVERVIEW_TILE_DEFAULT
        self._tile_spacing = _OVERVIEW_TILE_SPACING
        self._tile_sliders = []
        self._tile_size_changed = None
        self._measurement_settings = None
        self._comparison_requests = 0
        self._controls_lost = False
        self._plate_painted = True
        self._plate_built_at = 0.0
        self._original_steps = None
        self._original_generation = 0
        self.measurement_progress.connect(self._show_measurement_progress)
        self._geometry_valid = True
        self._geometry_error = None
        self._plate_cell_extent = None
        self._plate_fit = True
        self._collapsed = {}
        self._color_mode = "additive"
        self._custom_colors = {}
        self._tile_spacing_changed = None
        self._tile_spacing_boxes = []
        self._settings_flush = None
        self._image_menu_shown = None
        self._channel_limits = {}
        self._image_overrides = {}
        self._normalization_scopes = {"*": "image"}
        self._scope_cache = {}
        self._scope_generation = 0
        self._image_settings = None
        self._ink_intensity = 1.
        self._enhance_values = {}
        self._display_draft = {}
        self._global_display = {}
        self._write_display_rows = None
        self._view_state = _overview_empty_view_state()
        self._view_state_pending = None
        self._view_state_writing = False
        self._main_view_field = None
        self._displayed_field = None
        self._well_viewport = None
        self._readout_pending = None
        self._readout_planes = None
        self._readout_loading = None
        self.setWindowTitle(tr("Field-of-view overview"))
        self.resize(1000, 760)
        self._fields, self._visible = (), ()
        self._selected_fields = {}
        self._cache = OrderedDict()
        self._thumbnail_cache = OrderedDict()
        self._thumbnail_inflight = set()
        self._thumbnail_first_inflight = set()
        self._thumbnail_installs = []
        self._well_representatives = {}
        self._well_positions = {}
        self._installed_thumbnails = set()
        self._individual = False
        self._visual_signature = None
        self._displayed_montage = None
        self._montage_labels = False
        self._cancelled = threading.Event()
        self._index_jobs = JobRunner(self, app_key="field_overview", user_visible=False)
        self._index_jobs.job_finished.connect(self._finish_display_prepare)
        self._image_jobs = JobRunner(self, app_key="field_overview", user_visible=False)
        self._export_jobs = JobRunner(self, app_key="field_overview")
        self._thumbnail_jobs = JobRunner(self, app_key="field_overview", user_visible=False)
        self._montage_jobs = JobRunner(self, app_key="field_overview", user_visible=False)
        self._metadata_jobs = JobRunner(self, app_key="field_overview", user_visible=False)
        self._measurement_jobs = JobRunner(self, app_key="field_overview")
        self._native_jobs = JobRunner(self, app_key="field_overview", user_visible=False)
        self._native_timer = QTimer(self)
        self._native_timer.setSingleShot(True)
        self._native_timer.setInterval(120)
        self._native_timer.timeout.connect(self._load_native_detail)
        self._native_jobs.job_finished.connect(self._native_finished)
        self._field_thumbnail_jobs = JobRunner(self, app_key="field_overview", user_visible=False)
        self._field_thumbnail_jobs.job_finished.connect(self._request_visuals)
        self._visual_timer = QTimer(self)
        self._visual_timer.setSingleShot(True)
        self._visual_timer.timeout.connect(self._update_visuals)
        self._prepare_timer = QTimer(self)
        self._prepare_timer.setSingleShot(True)
        self._prepare_timer.timeout.connect(self._prepare_display_after_paint)
        self._install_timer = QTimer(self)
        self._install_timer.setSingleShot(True)
        self._install_timer.timeout.connect(self._install_pending_thumbnails)
        self._thumbnail_jobs.job_finished.connect(self._request_visuals)
        self._montage_jobs.job_finished.connect(self._request_visuals)
        self._scope = (None, None)
        self._image_jobs.job_finished.connect(self._refresh_image_if_pending)
        self._image_pending = False
        self._page = 0
        self._comparison_signature = None
        self._export_dock = None
        self._export_settings_state = None
        column = QVBoxLayout(self)
        source_row = QHBoxLayout()
        source = (settings or {}).get("source") or (settings or {}).get("src") or ""
        if isinstance(source, (list, tuple)):
            source = source[0] if source else ""
        self.source = QLineEdit(str(source))
        browse = QPushButton(tr("Browse folder"))
        browse.clicked.connect(self._browse)
        source_row.addWidget(self.source, 1)
        source_row.addWidget(browse)
        column.addLayout(source_row)
        self.regex = QLineEdit(str((settings or {}).get("custom_regex") or ""))
        self.regex.setPlaceholderText(tr("Filename regex with plateID, wellID, fieldID and chanID groups (optional)"))
        column.addWidget(self.regex)
        controls = QHBoxLayout()
        channels = (settings or {}).get("channels")
        if not isinstance(channels, (list, tuple)) or not channels:
            channels = (0,)
        self.channels = QLineEdit(",".join(str(value) for value in channels))
        self.channels.setPlaceholderText(tr("Channels, comma-separated"))
        self.channels.setToolTip(tr("Use filename channel IDs for separate files; use zero-based indices for multichannel images."))
        self.channels.setMaximumWidth(160)
        self.normalise = Toggle(tr("Percentile normalization"))
        self.normalise.setChecked(bool((settings or {}).get("normalize", True)))
        self.low, self.high = QDoubleSpinBox(), QDoubleSpinBox()
        percentiles = ((settings or {}).get("normalization") or {}).get("percentiles")
        percentiles = percentiles or (settings or {}).get("normalization_percentiles") or (1., 99.9)
        for spin, value in ((self.low, percentiles[0]), (self.high, percentiles[1])):
            spin.setRange(0, 100)
            spin.setValue(value)
        self.low.setToolTip(tr("Lower normalization percentile"))
        self.high.setToolTip(tr("Upper normalization percentile"))
        scan = QPushButton(tr("Load overview"))
        scan.clicked.connect(self._scan)
        metadata = QPushButton(tr("Filename metadata…"))
        metadata.clicked.connect(self._metadata)
        channel_pair = QWidget(self)
        channel_layout = QHBoxLayout(channel_pair)
        channel_layout.setContentsMargins(0, 0, 0, 0)
        channel_layout.addWidget(QLabel(tr("Channels")))
        channel_layout.addWidget(self.channels)
        controls.addWidget(channel_pair)
        for widget in (self.normalise, self.low, self.high, scan, metadata):
            controls.addWidget(widget)
        column.addLayout(controls)
        self.plates, self.rows, self.columns = QComboBox(), QComboBox(), QComboBox()
        navigation = QHBoxLayout()
        for widget in (self.plates, self.rows, self.columns):
            navigation.addWidget(widget)
            widget.currentIndexChanged.connect(self._filter)
        back = QPushButton(tr("Back to plate"))
        back.clicked.connect(self._back)
        navigation.addWidget(back)
        open_well = QPushButton(tr("Open selected well"))
        open_well.clicked.connect(self._open_selected_well)
        navigation.addWidget(open_well)
        self._plate_back_button, self._open_well_button = back, open_well
        self._navigation_layout = navigation
        column.addLayout(navigation)
        self._plates_view = "single"
        self._plates_subset = None
        self._plate_tables = {}
        self._plate_blocks = {}
        self._plate_extents = {}
        self._plate_available = {}
        self._readout_viewports = {}
        self._plates_host = QWidget()
        self._plates_host.setObjectName("FieldOverviewPlates")
        self._plates_layout = QBoxLayout(QBoxLayout.TopToBottom, self._plates_host)
        self._plates_layout.setContentsMargins(0, 0, 0, 0)
        self._plates_layout.setSpacing(6)
        self.wells = self._new_plate_table()
        self.fields = self._new_field_list()
        self._well_viewport = self.wells.viewport()
        from PySide6.QtWidgets import QSplitter
        from .ai_toggle_label import AiToggleLabel
        from ..theme import ensure_widget_qss_applied, register_widget_qss

        register_widget_qss("FieldOverviewPanel", _overview_panel_qss, replace=True)
        self._plate_panel = _OverviewPanel(tr("Plate"))
        self._plate_panel.body_layout.addWidget(self._plates_host)
        self._levels = QSplitter(Qt.Vertical)
        self._levels.setChildrenCollapsible(False)
        self._levels.addWidget(self._plate_panel)
        self._well_panels = []
        self._well_panel = self._new_well_panel(self.fields)
        self._levels.addWidget(self._well_panel)
        self.view = _OverviewImageView()
        self.view.axis_step.connect(self._step_axis)
        self.view.detail_requested.connect(self._request_detail)
        self.view.horizontalScrollBar().valueChanged.connect(self._request_native_detail)
        self.view.verticalScrollBar().valueChanged.connect(self._request_native_detail)
        self._image_panel = _OverviewPanel(tr("Image"), shrink_width=True)
        self._image_panel.body_layout.addWidget(self.view)
        self.metadata_toggle = AiToggleLabel(text="metadata", tooltip=_OVERVIEW_METADATA_TOOLTIP)
        self.metadata_toggle.toggled.connect(self._image_metadata_toggled)
        self._image_panel.header_layout.addWidget(self.metadata_toggle)
        self._image_metadata = QLabel(self.view.viewport())
        self._image_metadata.setObjectName("FieldOverviewImageMetadata")
        self._image_metadata.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._image_metadata.setWordWrap(True)
        self._image_metadata.hide()
        self._image_panel.hide()
        self._image_views = []
        self._image_stack_jobs = JobRunner(self, app_key="field_overview", user_visible=False)
        self._image_stack = QSplitter(Qt.Vertical)
        self._image_stack.setObjectName("FieldOverviewImageStack")
        self._image_stack.setChildrenCollapsible(False)
        image_column = QWidget()
        image_layout = QVBoxLayout(image_column)
        image_layout.setContentsMargins(0, 0, 0, 0)
        from .sortable_table import install_sorting

        self.comparison_table = QTableWidget()
        self.comparison_table.setObjectName("FieldOverviewComparisonTable")
        self.comparison_table.setToolTip(tr("Pairwise differences between the compared groups; adjusted p-values and effect sizes."))
        self.comparison_table.setMaximumHeight(180)
        self.comparison_table.setEditTriggers(QTableWidget.NoEditTriggers)
        install_sorting(self.comparison_table)
        self.comparison_table.hide()
        image_layout.addWidget(self.comparison_table)
        image_layout.addWidget(self._image_stack, 1)
        self._image_stack.addWidget(self._image_panel)
        self._image_column = image_column
        image_column.hide()
        splitter = QSplitter(Qt.Horizontal)
        self._splitter = splitter
        splitter.addWidget(self._levels)
        splitter.addWidget(image_column)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        column.addWidget(splitter, 1)
        ensure_widget_qss_applied("FieldOverviewPanel", root=self)
        axes = QHBoxLayout()
        self.fov_axis, self.time_axis, self.z_axis = QComboBox(), QComboBox(), QComboBox()
        self.z_mode = QComboBox()
        for text, value in ((tr("Z plane"), "plane"), (tr("Maximum Z projection"), "max"),
                            (tr("Mean Z projection"), "mean")):
            self.z_mode.addItem(text, value)
        self.z_mode.setCurrentIndex(max(0, self.z_mode.findData(self._overview_options.get("z_mode", "plane"))))
        self.wheel_axis = QComboBox()
        for text, value in ((tr("Wheel: zoom"), "zoom"), (tr("Wheel: field"), "field"),
                            (tr("Wheel: Z"), "z"), (tr("Wheel: time"), "time")):
            self.wheel_axis.addItem(text, value)
        self.wheel_axis.currentIndexChanged.connect(self._wheel_mode_changed)
        for label, combo in ((tr("Field"), self.fov_axis), (tr("Time"), self.time_axis),
                             (tr("Z"), self.z_axis)):
            axes.addWidget(QLabel(label))
            axes.addWidget(combo)
            combo.currentIndexChanged.connect(self._axis_changed)
        axes.addWidget(self.z_mode)
        self.z_mode.currentIndexChanged.connect(self._axis_changed)
        axes.addWidget(self.wheel_axis)
        self.preview_resolution = QComboBox()
        self.preview_resolution.setObjectName("OverviewPreviewResolution")
        for value in _OVERVIEW_QUALITY_STEPS:
            self.preview_resolution.addItem(str(value) if value else tr("Original"), value)
        self.preview_resolution.setCurrentIndex(3)
        self.preview_resolution.setToolTip(tr(
            "Maximum preview edge in pixels; lower values load faster. Zoom still loads original pixels."))
        self._quality_timer = QTimer(self)
        self._quality_timer.setSingleShot(True)
        self._quality_timer.setInterval(_OVERVIEW_QUALITY_DEBOUNCE_MS)
        self._quality_timer.timeout.connect(self._apply_resolution)
        self.preview_resolution.currentIndexChanged.connect(self._resolution_changed)
        axes.addWidget(QLabel(tr("Preview resolution")))
        axes.addWidget(self.preview_resolution)
        self.resolution_slider = self._quality_slider("FieldOverviewImageResolution")
        self.plate_resolution_slider = self._quality_slider("FieldOverviewPlateResolution")
        self.preview_resolution.currentIndexChanged.connect(self._sync_resolution_slider)
        header = self._image_panel.header_layout
        header.insertWidget(header.indexOf(self.metadata_toggle), self.resolution_slider)
        self._plate_panel.header_layout.addWidget(self.plate_resolution_slider)
        compare = QPushButton(tr("Compare selected fields"))
        compare.clicked.connect(self._compare)
        axes.addWidget(compare)
        self._compare_fields_button = compare
        open_images = QPushButton(tr("Open selected images"))
        open_images.setToolTip(tr("Open each selected field as its own enlarged image, stacked to compare them."))
        open_images.clicked.connect(self._open_selected_images)
        axes.addWidget(open_images)
        self._open_images_button = open_images
        self._axes_layout = axes
        column.addLayout(axes)
        annotations = QHBoxLayout()
        from .setup_dialog import _SetupComboBox

        self.annotation_scope = _SetupComboBox()
        for text, value in ((tr("Plate"), "plate"), (tr("Row"), "row"),
                            (tr("Column"), "column"), (tr("Selected wells"), "wells")):
            self.annotation_scope.addItem(text, value)
        self.annotation_scope.setToolTip(tr(
            "Apply metadata in this scope; an empty value removes the selected metadata item."))
        self.annotation_key = _SetupComboBox()
        self.annotation_key.setEditable(True)
        for label, key in ((tr("Condition"), "condition"), (tr("Cell type"), "cell_type"),
                           (tr("Pathogen"), "pathogen"), (tr("Genotype"), "genotype"), (tr("Drug"), "drug")):
            self.annotation_key.addItem(label, key)
        self.annotation_text = QLineEdit()
        self.annotation_text.setPlaceholderText(tr("Value"))
        annotate = QPushButton(tr("Apply annotation"))
        annotate.clicked.connect(self._annotate)
        save_annotations = QPushButton(tr("Save annotations"))
        load_annotations = QPushButton(tr("Load annotations"))
        save_annotations.clicked.connect(self._save_annotations)
        load_annotations.clicked.connect(self._load_annotations)
        self._annotation_buttons = (annotate, save_annotations, load_annotations)
        for widget in (self.annotation_scope, QLabel(tr("Metadata")), self.annotation_key,
                       QLabel(tr("Value")), self.annotation_text, annotate,
                       save_annotations, load_annotations):
            annotations.addWidget(widget)
        column.addLayout(annotations)
        quality = QHBoxLayout()
        self.representatives = QSpinBox()
        self.representatives.setRange(1, 24)
        self.representatives.setToolTip(tr("Display up to 24 representative fields per well; whole-source exports remain available."))
        self.representatives.setValue(int(self._overview_options.get("representatives", 1)))
        self.representatives.valueChanged.connect(self._representatives_changed)
        self.export_scope = _SetupComboBox()
        for text, value in ((tr("Current view"), "view"), (tr("Whole plate"), "plate"),
                            (tr("Current well"), "well"), (tr("Current field"), "field"),
                            (tr("Selected fields or wells"), "selection")):
            self.export_scope.addItem(text, value)
        self.export_scope.setToolTip(tr(
            "Whole plate and well exports use the selected Z and time; explicit selections keep their own acquisition identities."))
        self.pdf_quality, self.pdf_dpi = _SetupComboBox(), _SetupComboBox()
        for text, value in ((tr("PDF: full source resolution"), None),
                            (tr("PDF: 1600-pixel field"), 1600), (tr("PDF: 3200-pixel field"), 3200)):
            self.pdf_quality.addItem(text, value)
        for value in (150, 300, 600):
            self.pdf_dpi.addItem(tr("Resolution: {dpi} DPI", dpi=value), value)
        self.pdf_dpi.setCurrentIndex(1)
        for widget in (QLabel(tr("Representatives per well")), self.representatives,
                       self.export_scope, self.pdf_quality, self.pdf_dpi):
            quality.addWidget(widget)
        column.addLayout(quality)
        paging = QHBoxLayout()
        previous, following = QPushButton(tr("Previous fields")), QPushButton(tr("Next fields"))
        previous.clicked.connect(lambda: self._turn(-1))
        following.clicked.connect(lambda: self._turn(1))
        self.grouping = _SetupComboBox()
        for text, key in ((tr("Plate"), "plate"), (tr("Row"), "row"),
                          (tr("Column"), "column"), (tr("Well"), "well")):
            self.grouping.addItem(text, key)
        self.count, self.seed = QSpinBox(), QSpinBox()
        self.count.setRange(1, 1000)
        self.count.setValue(1)
        self.seed.setRange(0, 2147483647)
        self.random = Toggle(tr("Random sample"))
        self.use_selection = Toggle(tr("Export selected fields or wells"))
        self.sample_selected = Toggle(tr("Sample within selected wells"))
        self.measurements = Toggle(tr("Measurements CSV (raw image pixels)"))
        self.measurements.setToolTip(tr(
            "Measure original-resolution selected channels as one whole-image ROI, not segmented objects. "
            "Geometry is in pixels; intensity statistics use finite stored pixel values without display "
            "normalization. Nonfinite pixel counts and read errors are recorded."))
        export, cancel = QPushButton(tr("Contact sheet PDFs")), QPushButton(tr("Cancel loading or export"))
        export.setToolTip(tr("Twelve fields per page from the export scope, with the display settings printed above them."))
        exports = QPushButton(tr("Export…"))
        exports.setObjectName("OverviewExportMenuButton")
        exports.setToolTip(tr("Guided export, Custom export or contact sheet PDFs."))
        exports.clicked.connect(functools.partial(self._export_menu, exports))
        clear_selection = QPushButton(tr("Clear selections"))
        clear_selection.clicked.connect(self._clear_selection)
        export.clicked.connect(self._export)
        cancel.clicked.connect(self._cancel)
        for widget in (previous, following):
            paging.addWidget(widget)
        self._paging_buttons = (previous, following)
        self._clear_selection_button = clear_selection
        self._plate_scale = 1.
        self._plate_zoom_buttons = []
        for text, step in ((tr("Fit plate"), 0), (tr("Larger wells"), 1), (tr("Smaller wells"), -1)):
            button = QPushButton(text)
            button.clicked.connect(lambda _checked=False, value=step: self._zoom_plate(value))
            paging.addWidget(button)
            self._plate_zoom_buttons.append(button)
        self._plate_buttons = []
        for text, tooltip, handler in (
                (tr("Previous plate"), "", functools.partial(self._step_plate, -1)),
                (tr("Next plate"), "", functools.partial(self._step_plate, 1)),
                (tr("Add plates…"), tr("Add plate folders, or folders of plates, to this Overview."),
                 self._choose_plates),
                (tr("Remove plate"), tr("Remove the shown plate from this Overview; its files are not touched."),
                 self._remove_shown_plate)):
            button = QPushButton(text)
            button.setToolTip(tooltip)
            button.clicked.connect(handler)
            paging.addWidget(button)
            self._plate_buttons.append(button)
        from PySide6.QtWidgets import QMenu

        self.plates_view = QComboBox()
        self.plates_view.setObjectName("FieldOverviewPlatesView")
        for value, caption in _OVERVIEW_PLATES_VIEWS:
            self.plates_view.addItem(tr(caption), value)
        self.plates_view.setToolTip(tr(_OVERVIEW_PLATES_VIEW_TOOLTIP))
        self.plates_view.setAccessibleName(tr("Plates view"))
        self.plates_view.currentIndexChanged.connect(self._plates_view_chosen)
        shown_plates = QPushButton(tr("Shown plates…"))
        shown_plates.setObjectName("FieldOverviewShownPlates")
        shown_plates.setToolTip(tr("Choose which loaded plates Side by side and Stacked show."))
        self._shown_plates_menu = QMenu(shown_plates)
        self._shown_plates_menu.aboutToShow.connect(self._fill_shown_plates_menu)
        shown_plates.setMenu(self._shown_plates_menu)
        self._plates_view_controls = (self.plates_view, shown_plates)
        for widget in self._plates_view_controls:
            paging.addWidget(widget)
        column.addLayout(paging)
        sample_row = QHBoxLayout()
        for widget in (self.grouping, QLabel(tr("Samples per group")), self.count,
                       QLabel(tr("Seed")), self.seed, self.random):
            sample_row.addWidget(widget)
        column.addLayout(sample_row)
        export_row = QHBoxLayout()
        for widget in (self.use_selection, self.sample_selected, clear_selection, exports, export, cancel):
            export_row.addWidget(widget)
        column.addLayout(export_row)
        column.addWidget(self.measurements)
        self.status = QLabel(tr("No images have been loaded."))
        self.status.setWordWrap(True)
        column.addWidget(self.status)
        for runner in (self._index_jobs, self._image_jobs, self._export_jobs,
                       self._metadata_jobs, self._measurement_jobs, self._native_jobs):
            runner.job_failed.connect(self.status.setText)
        for control in (self.channels,):
            control.editingFinished.connect(self._refresh_image)
        self.normalise.toggled.connect(self._refresh_image)
        self.low.valueChanged.connect(self._refresh_image)
        self.high.valueChanged.connect(self._refresh_image)
        self._convention = str((settings or {}).get("metadata_type") or "auto")
        self._build_advanced_controls(column)
        self._loading_controls = False
        self._normalization_control_states()
        from ..screens.settings_model import retarget_field_tooltips
        retarget_field_tooltips(self)
        self._wrap_controls(column, splitter)
        from .flow import FlowHost, FlowLayout
        from PySide6.QtWidgets import QScrollArea
        self._module_actions = QScrollArea(self)
        self._module_actions.setWidgetResizable(True)
        self._module_actions.setFrameShape(QScrollArea.NoFrame)
        self._module_actions.viewport().setAutoFillBackground(False)
        self._module_action_content = FlowHost()
        action_row = FlowLayout(self._module_action_content)
        self._module_actions.setWidget(self._module_action_content)
        self._module_action_buttons = []
        for caption, callback in ((tr("Export image-level measurements"), self._measure),
                (tr("Export…"), self._export_menu),
                (tr("Cancel loading or export"), self._cancel)):
            button = QPushButton(caption)
            button.clicked.connect(functools.partial(callback, button) if callback == self._export_menu
                                   else callback)
            action_row.addWidget(button)
            self._module_action_buttons.append(button)
        self._module_actions_retired = False
        self._fitting_module_actions = False
        self._module_action_fit_timer = QTimer(self)
        self._module_action_fit_timer.setSingleShot(True)
        self._module_action_fit_timer.timeout.connect(self._fit_module_actions)
        for widget in (self._module_actions, self._module_actions.viewport(),
                       self._module_action_content, *self._module_action_buttons):
            widget.installEventFilter(self)
        self._module_action_fit_timer.start(0)
        self._module_actions.hide()
        self._build_pixel_readout(column)
        self._build_image_settings()

    def _build_pixel_readout(self, column):
        """Add the fixed-height strip that shows raw pixel values under the cursor.

        :param column: the browser's main layout; the strip goes at its bottom left.
        """
        self._pixel_readout = QLabel()
        self._pixel_readout.setObjectName("FieldOverviewPixelReadout")
        self._pixel_readout.setTextFormat(Qt.PlainText)
        self._pixel_readout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._pixel_readout.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self._pixel_readout.setFixedHeight(self._pixel_readout.fontMetrics().height() + 8)
        self._pixel_readout.setAccessibleName(tr("Pixel values under the cursor"))
        column.addWidget(self._pixel_readout)
        self._readout_timer = QTimer(self)
        self._readout_timer.setSingleShot(True)
        self._readout_timer.setInterval(_OVERVIEW_READOUT_INTERVAL_MS)
        self._readout_timer.timeout.connect(self._update_pixel_readout)
        self._readout_jobs = JobRunner(self, app_key="field_overview", user_visible=False)
        self._track_readout(self.view.viewport(), self.view)

    def _build_image_settings(self):
        """Prepare the right-click image settings: the badge, the menu and the debounced save."""
        self._levels_jobs = JobRunner(self, app_key="field_overview", user_visible=False)
        self._override_save_timer = QTimer(self)
        self._override_save_timer.setSingleShot(True)
        self._override_save_timer.setInterval(400)
        self._override_save_timer.timeout.connect(self._persist_view_state)
        self._image_adjusted_badge = self._adjusted_badge()
        header = self._image_panel.header_layout
        header.insertWidget(header.indexOf(self.resolution_slider), self._image_adjusted_badge)
        self.view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.view.customContextMenuRequested.connect(functools.partial(self._image_menu, self.view))

    def _track_readout(self, viewport, owner):
        """Show the pixel readout while the cursor moves over ``owner``'s images.

        :param viewport: the viewport that receives the mouse moves.
        :param owner: the image view or well view that draws the images.
        """
        viewport.setMouseTracking(True)
        viewport.installEventFilter(self)
        self._readout_viewports[viewport] = owner

    def _readout_event(self, owner, event):
        """Throttle cursor moves over an image and clear the readout when it leaves.

        :param owner: the image view or well view under the cursor.
        :param event: the mouse move or leave event.
        """
        if event.type() == QEvent.Leave:
            self._clear_pixel_readout()
            return
        self._readout_pending = (owner, event.position())
        if not self._readout_timer.isActive():
            self._readout_timer.start()

    def _clear_pixel_readout(self):
        """Empty the readout strip; its height stays, so nothing moves."""
        self._readout_pending = None
        if getattr(self, "_pixel_readout", None) is None:
            return
        self._readout_timer.stop()
        self._pixel_readout.clear()

    def _readout_target(self, owner, position):
        """Find the field and rendered-image position under the cursor.

        :param owner: the image view or well view under the cursor.
        :param position: the cursor in the owner's viewport, in logical pixels.
        :returns: ``(field, u, v, (rendered height, rendered width))`` with
            ``u, v`` in rendered image pixels, or ``None`` off every image.
        """
        from PySide6.QtCore import QRectF

        if isinstance(owner, ZoomableImageView):
            item = owner._item
            if item is None or item.pixmap().isNull():
                return None
            field = self._field_shown_in(owner)
            if field is None or (owner is self.view and not self._individual):
                return None
            inverse, invertible = owner.viewportTransform().inverted()
            if not invertible:
                return None
            local = item.mapFromScene(inverse.map(position))
            pixmap = item.pixmap()
            ratio = pixmap.devicePixelRatio()
            return field, local.x() * ratio, local.y() * ratio, (pixmap.height(), pixmap.width())
        item = self._tile_at(owner, position.toPoint())
        rendered = item.data(Qt.UserRole + 3) if item is not None else None
        if not rendered or item.icon().isNull():
            return None
        option = QStyleOptionViewItem()
        option.initFrom(owner.viewport())
        option.rect = owner.visualItemRect(item)
        option.decorationSize = owner.iconSize()
        option.decorationPosition = QStyleOptionViewItem.Top
        option.decorationAlignment = Qt.AlignCenter
        option.displayAlignment = Qt.AlignHCenter | Qt.AlignTop
        option.features = QStyleOptionViewItem.HasDecoration | (
            QStyleOptionViewItem.HasDisplay if item.text() else QStyleOptionViewItem.None_)
        option.text = item.text()
        option.icon = item.icon()
        decoration = owner.style().subElementRect(QStyle.SE_ItemViewItemDecoration, option, owner)
        size = item.icon().actualSize(owner.iconSize())
        shown = QRectF(decoration.x() + (decoration.width() - size.width()) / 2,
                       decoration.y() + (decoration.height() - size.height()) / 2,
                       size.width(), size.height())
        if shown.isEmpty() or not shown.contains(position):
            return None
        height, width = rendered
        return (item.data(Qt.UserRole), (position.x() - shown.x()) / shown.width() * width,
                (position.y() - shown.y()) / shown.height() * height, (height, width))

    def _update_pixel_readout(self):
        """Show the raw value of every channel at the source pixel under the cursor."""
        if self._readout_pending is None or self._cancelled.is_set():
            return
        owner, position = self._readout_pending
        if owner in self._plate_blocks:
            self._pixel_readout.setText(self._plate_hover_text(owner, position))
            return
        target = self._readout_target(owner, position)
        if target is None:
            self._pixel_readout.clear()
            return
        field, u, v, rendered = target
        axes = self._axis_options()
        key = (field.identity, tuple(sorted(axes.items())))
        place = "   ·   ".join((tr("Plate {plate}", plate=field.plate),
                                tr("Well {well}", well=field.well or tr("Unresolved")),
                                tr("Image {image}", image=field.field)))
        if self._readout_planes is None or self._readout_planes[0] != key:
            self._load_readout_planes(field, key, axes)
            self._pixel_readout.setText(place + "   ·   " + tr("reading pixel values…"))
            return
        planes = self._readout_planes[1]
        if isinstance(planes, str):
            self._pixel_readout.setText(place + "   ·   " + planes)
            return
        try:
            channels = self._choices()[0]
        except (ValueError, TypeError):
            channels = ()
        mosaic = self._mosaic(channels)
        shape = next(iter(planes.values())).shape[:2]
        point = _overview_source_pixel(u, v, rendered, shape, len(channels), mosaic)
        if point is None:
            self._pixel_readout.clear()
            return
        x, y = point
        names = self._overview_options.get("channel_names") or {}
        values = []
        for channel, plane in planes.items():
            name = names.get(channel, names.get(str(channel), "")) or tr("Channel {channel}", channel=channel)
            values.append(f"{name}: {_overview_raw_value(plane[y, x])}")
        self._pixel_readout.setText("   ·   ".join((tr("x {x}, y {y}", x=x, y=y), place, *values)))

    def _plate_hover_text(self, table, position):
        """Name the plate, well and shown image under the cursor on a plate table.

        :param table: the hovered plate table.
        :param position: the cursor in its viewport.
        :returns: the readout text, or ``""`` off every well.
        """
        plate = getattr(table, "_overview_plate", None)
        item = table.itemAt(position.toPoint())
        if plate is None or item is None:
            return ""
        index = self.plates.findData(plate)
        parts = [tr("Plate {plate}", plate=self.plates.itemText(index) if index >= 0 else plate)]
        well = item.data(Qt.UserRole)
        if well is not None:
            parts.append(tr("Well {well}", well=well or tr("Unresolved")))
            field = self._well_representatives.get((plate, well))
            if field is not None:
                parts.append(tr("Image {image}", image=field.field))
        return "   ·   ".join(parts)

    def _load_readout_planes(self, field, key, axes):
        """Read every channel of ``field`` at original resolution for the readout, once.

        :param field: the hovered field.
        :param key: the field identity and acquisition axes the planes belong to.
        :param axes: the time, Z and projection the views show.
        """
        if self._readout_loading == key:
            return
        self._readout_loading = key
        self._readout_jobs.submit(functools.partial(_overview_readout_planes, field, axes),
                                  functools.partial(self._readout_planes_read, key, self._cancelled))

    def _readout_planes_read(self, key, cancelled, planes):
        """Keep the read planes and refresh the readout if the cursor is still on an image.

        :param key: the field identity and axes the planes belong to.
        :param cancelled: the source generation the read started in.
        :param planes: ``{channel: raw plane}``, or the reason they are missing.
        """
        if self._readout_loading == key:
            self._readout_loading = None
        if cancelled.is_set():
            return
        self._readout_planes = (key, planes)
        if self._readout_pending is not None:
            self._update_pixel_readout()

    def _tile_at(self, fields, point):
        """The well-view tile whose cell holds ``point``, or ``None``.

        A tile's whole grid cell counts, not only the part Qt's own hit test
        reaches, so the lower half of a tall thumbnail answers too.

        :param fields: the well view.
        :param point: a point in its viewport.
        """
        return next((fields.item(index) for index in range(fields.count())
                     if fields.visualItemRect(fields.item(index)).contains(point)), None)

    def _is_adjusted(self, field):
        """Whether ``field`` has its own display settings."""
        return _overview_image_key(field) in self._image_overrides

    def _adjusted_badge(self):
        """A small "Adjusted" badge for an image panel's header, hidden until needed."""
        badge = QLabel(tr("Adjusted"))
        badge.setObjectName("FieldOverviewAdjustedBadge")
        badge.setToolTip(tr("This image has its own display settings; right-click it to change or reset them."))
        badge.hide()
        return badge

    def _refresh_adjusted_badges(self):
        """Show the badge on every open image with its own settings and repaint the well views."""
        main = self._main_view_field
        self._image_adjusted_badge.setVisible(
            bool(self._individual and main is not None and self._is_adjusted(main)))
        for panel in self._image_views:
            panel.adjusted_badge.setVisible(self._is_adjusted(panel.field))
        for fields in self._field_lists():
            fields.viewport().update()

    def _set_image_override(self, key, override):
        """Replace one image's display override, redraw every view and save it soon.

        :param key: the image key.
        :param override: the new override, or ``None`` to reset the image.
        """
        overrides = dict(self._image_overrides)
        cleaned = _overview_clean_override(override)
        if cleaned is None:
            overrides.pop(key, None)
        else:
            overrides[key] = cleaned
        if overrides == self._image_overrides:
            return
        self._set_display_layers(image_overrides=overrides)
        self._refresh_adjusted_badges()
        self._override_save_timer.start()

    def _copy_image_override(self, field, scope, channel=None):
        """Copy ``field``'s settings to the other images of its well, its plate or every image.

        :param field: the image copied from.
        :param scope: ``'well'``, ``'plate'`` or ``'all'``.
        :param channel: copy only this channel, or ``None`` for every channel.
        """
        key = _overview_image_key(field)
        targets = _overview_override_targets(self._fields, field, scope)
        overrides = _overview_copied_overrides(self._image_overrides, key, targets, channel)
        if overrides != self._image_overrides:
            self._set_display_layers(image_overrides=overrides)
            self._refresh_adjusted_badges()
            self._override_save_timer.start()
        self.status.setText(tr("Display settings copied to {count} images.", count=max(0, len(targets) - 1)))

    def _open_image_settings(self, field):
        """Open the image settings panel on ``field``, reusing an open panel.

        :param field: the image to edit.
        """
        from shiboken6 import isValid

        try:
            channels = self._choices()[0]
        except (ValueError, TypeError) as error:
            self.status.setText(str(error))
            return
        panel = self._image_settings
        if panel is None or not isValid(panel):
            panel = self._image_settings = _OverviewImageSettings(self, field, channels)
        else:
            panel.show_field(field, channels)
        panel.show()
        panel.raise_()
        panel.activateWindow()

    def _request_image_histograms(self, panel):
        """Read the panel image's display histograms off the GUI thread.

        :param panel: the image settings panel.
        """
        try:
            limits = self._choices()[1]
        except (ValueError, TypeError):
            limits = None
        options = dict(self._display_kwargs(view=True))
        options.pop("image_overrides", None)
        layers = {name: False for name, _caption, _tooltip in _OVERVIEW_GLOBAL_LAYERS
                  if (self._image_overrides.get(panel.key) or {}).get(name) is False}
        if layers:
            options["image_overrides"] = {panel.key: layers}
        self._levels_jobs.submit(
            functools.partial(_overview_display_histograms, panel.field, panel.channels, limits, options),
            functools.partial(self._image_histograms_read, panel.key))

    def _image_histograms_read(self, key, histograms):
        """Give read histograms to the panel if it still shows that image.

        :param key: the image key they were read for.
        :param histograms: ``{channel: counts}``, or the reason they are missing.
        """
        from shiboken6 import isValid

        panel = self._image_settings
        if panel is not None and isValid(panel) and panel.key == key:
            panel.set_histograms(histograms)

    def _image_menu(self, view, position):
        """Offer the image's own settings on a right click in an image view.

        :param view: the enlarged image or a stacked image's view.
        :param position: the click in the view's viewport.
        """
        from PySide6.QtWidgets import QMenu

        field = self._field_shown_in(view)
        if field is None or (view is self.view and not self._individual):
            return
        menu = QMenu(view)
        menu.addAction(tr("Image settings…")).triggered.connect(
            functools.partial(self._open_image_settings, field))
        self._add_global_layer_actions(menu, field)
        reset = menu.addAction(tr("Reset image settings"))
        reset.setEnabled(self._is_adjusted(field))
        reset.triggered.connect(functools.partial(self._set_image_override, _overview_image_key(field), None))
        if any(panel.view is view for panel in self._image_views):
            menu.addSeparator()
            link = menu.addAction(tr("Link zoom"))
            link.setObjectName("OverviewImage_link_zoom")
            link.setCheckable(True)
            link.setChecked(self._link_zoom)
            link.setToolTip(tr(_OVERVIEW_LINK_ZOOM_TOOLTIP))
            link.toggled.connect(functools.partial(self._set_link_zoom, view))
        menu.popup(view.viewport().mapToGlobal(position))
        self._image_menu_shown = menu

    def _set_link_zoom(self, view, on):
        """Zoom and pan the stacked images together, or each on its own.

        :param view: the stacked image whose menu changed it; the others take
            its zoom and position when the link is switched on.
        :param on: whether the stacked images move together.
        """
        self._link_zoom = bool(on)
        if self._link_zoom:
            self._sync_image_views(view)

    def _add_global_layer_actions(self, menu, field):
        """Add the Global normalization and Global enhancement ticks of ``field`` to ``menu``.

        :param menu: the image's context menu.
        :param field: the image the ticks belong to.
        """
        menu.setToolTipsVisible(True)
        override = self._image_overrides.get(_overview_image_key(field)) or {}
        for name, caption, tooltip in _OVERVIEW_GLOBAL_LAYERS:
            action = menu.addAction(tr(caption))
            action.setObjectName("OverviewImage_" + name)
            action.setCheckable(True)
            action.setChecked(override.get(name) is not False)
            action.setToolTip(tr(tooltip))
            action.toggled.connect(functools.partial(self._set_global_layer, field, name))

    def _set_global_layer(self, field, name, on):
        """Draw ``field`` with or without one global layer and save that with its settings.

        :param field: the image.
        :param name: ``'global_normalization'`` or ``'global_enhancement'``.
        :param on: whether the global layer applies to this image.
        """
        import copy

        key = _overview_image_key(field)
        override = copy.deepcopy(self._image_overrides.get(key) or {})
        if on:
            override.pop(name, None)
        else:
            override[name] = False
        self._set_image_override(key, override)
        panel = self._image_settings
        from shiboken6 import isValid

        if panel is not None and isValid(panel) and panel.key == key:
            self._request_image_histograms(panel)

    def _fit_module_actions(self):
        """Refresh native action minima after coalesced font and style changes."""
        if self._module_actions_retired or self._fitting_module_actions:
            return
        self._fitting_module_actions = True
        try:
            from PySide6.QtWidgets import QApplication
            from ..gui_scale import current_scale
            from ..preferences import get_font_scale

            app = QApplication.instance()
            context = (current_scale(), get_font_scale(), app.font().toString(),
                       self.window().property("_spacr_window_stylesheet_serial"),
                       self._module_action_content.font().toString())
            width = (self._module_actions.viewport().width()
                     if self._module_actions.isVisibleTo(self.window()) else 0)
            for button in self._module_action_buttons:
                _fit_overview_action_width(button, width, (context, button.text()))
            if width > 0:
                from PySide6.QtCore import QSize
                from ..gui_scale import _resize_actual

                layout = self._module_action_content.layout()
                target_width = max(width, layout.minimumSize().width())
                target = QSize(target_width, max(self._module_actions.viewport().height(),
                                                layout.heightForWidth(target_width)))
                if self._module_action_content.property("size") != target:
                    layout.activate()
                    _resize_actual(self._module_action_content, target)
            self._fit_plate_allocation()
        finally:
            self._fitting_module_actions = False

    def _fit_plate_allocation(self):
        """Reserve the delegate's image budget and show every Figure setting.

        The delegate's 20-pixel image budget and cell insets are preferred
        only within the enclosing viewport capacity. Dense plates retain
        whole-plate Fit rather than imposing an unbounded per-row floor.
        The Figure settings flow, which lives in the settings column, is
        given the height its wrapped rows need at the column's width.
        A scrolling ancestor bounds surplus table stretch; removing that
        ancestor restores the unbounded maximum, and resizing recomputes it.
        Only embedded content receives its fresh layout's width-dependent
        floor; standalone content regains its original minimum height.
        """
        from ..gui_scale import _set_actual_dim
        from PySide6.QtWidgets import QScrollArea

        scroll = self._essential_scroll
        if not hasattr(self, "_allocation_minimum_baseline"):
            self._allocation_minimum_baseline = self.property("minimumHeight")
        content = scroll.widget()
        rows = content.heightForWidth(max(1, scroll.viewport().width())) + 2 * scroll.frameWidth()
        if content.minimumSizeHint().width() > scroll.viewport().width():
            rows += scroll.horizontalScrollBar().sizeHint().height()
        if scroll.property("minimumHeight") != rows:
            _set_actual_dim(scroll, "minh", rows)
            _set_actual_dim(scroll, "maxh", rows)
        available = self.window().property("size").height()
        ancestor = self.parentWidget()
        viewports = []
        while ancestor is not None:
            if isinstance(ancestor, QScrollArea):
                viewport = ancestor.viewport()
                viewports.append(viewport)
                available = min(available, viewport.property("size").height())
            ancestor = ancestor.parentWidget()
        for viewport in getattr(self, "_allocation_viewports", ()):
            if viewport not in viewports:
                viewport.removeEventFilter(self)
        for viewport in viewports:
            viewport.installEventFilter(self)
        self._allocation_viewports = tuple(viewports)
        if not viewports:
            available = min(available, self.property("size").height())
        budget = max(0, available)
        tables = self._plate_table_list()
        share = budget // len(tables) if self._plates_view == "stacked" else budget
        for table in tables:
            gaps = [table.rowHeight(index) - table.visualItemRect(item).height()
                    for index in range(table.rowCount())
                    if (item := table.item(index, 0)) is not None]
            gap = max(gaps, default=int(table.showGrid()))
            plate = (22 + gap) * max(1, table.rowCount()) + 2 * table.frameWidth()
            if table.horizontalHeader().isVisible():
                plate += table.horizontalHeader().property("size").height()
            plate = min(plate, share)
            if table.property("minimumHeight") != plate:
                _set_actual_dim(table, "minh", plate)
            maximum = share if viewports else 16777215
            if table.property("maximumHeight") != maximum:
                _set_actual_dim(table, "maxh", maximum)
        floor = self._allocation_minimum_baseline
        if viewports:
            body_layout = self.layout()
            floor = max(floor, body_layout.minimumSize().height(),
                        body_layout.totalHeightForWidth(self.property("size").width()))
        if self.property("minimumHeight") != floor:
            _set_actual_dim(self, "minh", floor)

    def _new_field_list(self):
        """Create one well's field thumbnails with the shared selection behaviour.

        :returns: an icon list whose clicks and keyboard moves activate its well.
        """
        from PySide6.QtCore import QSize
        from PySide6.QtWidgets import QAbstractItemView, QListView, QListWidget

        fields = QListWidget()
        fields.setSelectionMode(QAbstractItemView.ExtendedSelection)
        fields.setViewMode(QListView.IconMode)
        fields.setResizeMode(QListView.Adjust)
        fields.setMovement(QListView.Static)
        fields.setSelectionRectVisible(True)
        fields.setWrapping(True)
        fields.setWordWrap(True)
        fields.setSpacing(0)
        fields.viewport().installEventFilter(self)
        fields.verticalScrollBar().valueChanged.connect(self._request_visuals)
        fields.itemSelectionChanged.connect(self._remember_selection)
        fields.itemClicked.connect(lambda item, owner=fields: self._field_in(owner, item))
        fields.currentItemChanged.connect(
            lambda current, previous, owner=fields: self._current_field_in(owner, current, previous))
        fields._overview_well, fields._overview_page = None, 0
        self._track_readout(fields.viewport(), fields)
        fields.setItemDelegate(_OverviewTileDelegate(self, fields))
        fields._overview_metadata = False
        fields.setContextMenuPolicy(Qt.CustomContextMenu)
        fields.customContextMenuRequested.connect(
            lambda position, owner=fields: self._field_menu(owner, position))
        self._apply_tile_geometry(fields)
        return fields

    def _field_menu(self, fields, position):
        """Offer stacked enlargement of the selected fields on a right click.

        :param fields: the well view that was clicked.
        :param position: the click position in its viewport.
        """
        from PySide6.QtWidgets import QMenu

        menu = QMenu(fields)
        action = menu.addAction(tr("Open selected images"))
        action.setEnabled(bool(self._selected_fields))
        action.triggered.connect(self._open_selected_images)
        item = self._tile_at(fields, position)
        if item is not None:
            field = item.data(Qt.UserRole)
            menu.addSeparator()
            menu.addAction(tr("Image settings…")).triggered.connect(
                functools.partial(self._open_image_settings, field))
            self._add_global_layer_actions(menu, field)
            reset = menu.addAction(tr("Reset image settings"))
            reset.setEnabled(self._is_adjusted(field))
            reset.triggered.connect(functools.partial(self._set_image_override, _overview_image_key(field), None))
        menu.popup(fields.viewport().mapToGlobal(position))
        self._image_menu_shown = menu

    def _apply_tile_geometry(self, fields):
        """Size one well view's tiles from the tile size and spacing.

        The grid is the image plus the spacing, so neighbouring images sit
        close together; shown captions add two text lines below each image.

        :param fields: the well view's list.
        """
        from PySide6.QtCore import QSize

        edge = int(self._tile_size)
        height = max(1, round(edge * 3 / 4))
        text = 2 * fields.fontMetrics().height() if fields._overview_metadata else 0
        fields.setIconSize(QSize(edge, height))
        fields.setGridSize(QSize(edge + self._tile_spacing, height + self._tile_spacing + text))

    def _quality_slider(self, name):
        """Build one image-quality slider that moves the shared preview resolution.

        :param name: the slider's object name.
        :returns: a slider whose top position loads the original pixels.
        """
        from PySide6.QtWidgets import QSlider

        slider = QSlider(Qt.Horizontal)
        slider.setObjectName(name)
        slider.setRange(0, self.preview_resolution.count() - 1)
        slider.setPageStep(1)
        slider.setValue(self.preview_resolution.currentIndex())
        slider.setMaximumWidth(140)
        slider.setToolTip(tr(_OVERVIEW_QUALITY_TOOLTIP))
        slider.setAccessibleName(tr("Image quality"))
        slider.valueChanged.connect(self.preview_resolution.setCurrentIndex)
        self._quality_sliders.append(slider)
        return slider

    def _sync_resolution_slider(self, index):
        """Keep every image-quality slider on the chosen preview resolution."""
        for slider in self._quality_sliders:
            if slider.value() != index:
                slider.blockSignals(True)
                slider.setValue(index)
                slider.blockSignals(False)

    def _quality_original(self):
        """Whether the image-quality slider is at its top, the original pixels."""
        return not self.preview_resolution.currentData()

    def _quality_cap(self):
        """The longest preview edge the views load; the largest step at Original."""
        return self.preview_resolution.currentData() or _OVERVIEW_QUALITY_STEPS[-2]

    def _set_tile_size(self, edge):
        """Resize the images of every well view and reload them at that size.

        :param edge: the tile edge in screen pixels.
        """
        edge = max(_OVERVIEW_TILE_RANGE[0], min(_OVERVIEW_TILE_RANGE[1], int(edge)))
        if edge == self._tile_size:
            return
        self._tile_size = edge
        for slider in self._tile_sliders:
            if slider.value() != edge:
                slider.blockSignals(True)
                slider.setValue(edge)
                slider.blockSignals(False)
        for fields in self._field_lists():
            self._apply_tile_geometry(fields)
        for panel in self._image_views:
            panel.view.setMinimumHeight(self._image_view_height())
        if callable(self._tile_size_changed):
            self._tile_size_changed(edge)
        self._request_visuals()

    def _set_tile_spacing(self, spacing):
        """Change the gap between neighbouring well-view images.

        :param spacing: the gap in screen pixels.
        """
        spacing = max(0, min(_OVERVIEW_TILE_SPACING_MAX, int(spacing)))
        changed = spacing != self._tile_spacing
        self._tile_spacing = spacing
        for box in self._tile_spacing_boxes:
            if box.value() != spacing:
                box.blockSignals(True)
                box.setValue(spacing)
                box.blockSignals(False)
        for fields in self._field_lists():
            self._apply_tile_geometry(fields)
        if changed and callable(self._tile_spacing_changed):
            self._tile_spacing_changed(spacing)

    def _original_chain(self, options):
        """The chain for an image opened with Original, PSF and restoration included.

        :param options: the view's display options, holding its enhancement.
        :returns: the chain with Make Masks' full-resolution steps, or ``None``
            when neither PSF deconvolution nor restoration is switched on.
        """
        fields = self._original_steps() if callable(self._original_steps) else {}
        if not fields:
            return None
        from ..detect_chain import NO_CHAIN

        return (options.get("enhancement") or NO_CHAIN)._replace(**fields)

    def _original_steps_changed(self):
        """Reload Original images after the PSF or the restoration model changes."""
        self._original_generation += 1
        self._displayed_field = None
        item = self.fields.currentItem()
        if self._original_image and item is not None and self._individual:
            self._field(item, preserve_view=True)
        if self._original_image:
            self._reload_image_views()

    def _set_channel_targets(self, enhance=None, normalize=None):
        """Restrict enhancement and the percentile stretch to chosen channels.

        :param enhance: channels enhanced, or ``None`` for all.
        :param normalize: channels percentile-stretched, or ``None`` for all.
        """
        if (enhance, normalize) == (self._enhance_channels, self._normalize_channels):
            return
        self._enhance_channels, self._normalize_channels = enhance, normalize
        self._display_generation += 1
        self._displayed_montage = None
        self._refresh_image()

    def _set_color_mode(self, mode, custom_colors=None):
        """Combine the shown channels in one of the colour modes of :data:`_OVERVIEW_COLOR_MODES`.

        :param mode: a colour mode; anything else is ``'additive'``.
        :param custom_colors: ``{channel text: '#rrggbb'}`` for the custom
            mode, or ``None`` to keep the current ones.
        """
        mode = str(mode or "additive").lower()
        mode = mode if mode in _OVERVIEW_COLOR_MODES else "additive"
        colours = self._custom_colors if custom_colors is None else dict(custom_colors)
        if mode == self._color_mode and colours == self._custom_colors:
            return
        self._color_mode, self._custom_colors = mode, colours
        self._display_generation += 1
        self._cache.clear()
        self._thumbnail_cache.clear()
        self._displayed_montage = None
        self._displayed_field = None
        self._request_visuals()
        self._reload_image_views()
        item = self.fields.currentItem()
        if item is not None and self._individual:
            self._field(item, preserve_view=True)

    def _apply_display(self):
        """Redraw every view with the applied display settings.

        The plate thumbnails, every well view, every open image view and the
        enlarged image are drawn again, so an image that was already open
        shows the new normalization and enhancement too.
        """
        if self._controls_lost:
            return
        self._display_generation += 1
        self._cache.clear()
        self._thumbnail_cache.clear()
        for fields in self._field_lists():
            for index in range(fields.count()):
                fields.item(index).setData(Qt.UserRole + 1, None)
        self._displayed_montage = None
        self._displayed_field = None
        self._refresh_image()

    def _display_colors(self):
        """The channel colours the views use: the custom ones in the custom mode, else the metadata step's."""
        if self._color_mode == "custom":
            return dict(self._custom_colors)
        return self._overview_options.get("channel_colors") or None

    def _mosaic(self, channels):
        """Whether ``channels`` are drawn side by side rather than combined into one colour image."""
        return len(channels) > 1 and _overview_color_plan(channels, self._display_colors(), self._color_mode) is None

    def _set_normalization_scopes(self, scopes):
        """Normalize each channel over its image, well, plate or every loaded plate.

        :param scopes: ``{channel text or "*": scope}``, as
            :func:`_overview_channel_scopes` reads the setting.
        """
        scopes = dict(scopes or {"*": "image"})
        if scopes == self._normalization_scopes:
            return
        self._normalization_scopes = scopes
        self._refresh_scope_limits()

    def _scope_request(self):
        """The scoped channels, percentiles, axes and own channel percentiles the limits use now."""
        try:
            channels, limits = self._choices()
        except (ValueError, TypeError):
            return None
        scoped = {channel: _overview_scope_of(self._normalization_scopes, channel) for channel in channels}
        scoped = {channel: scope for channel, scope in scoped.items() if scope != "image"}
        if not scoped or not self._fields:
            return None
        options = self._display_kwargs()
        axes = {key: options[key] for key in ("time_index", "z_mode", "z_index")}
        own = {channel: tuple(value) for channel, value in (options.get("channel_percentiles") or {}).items()
               if channel in scoped}
        return scoped, tuple(limits) if limits is not None else (0., 100.), axes, own

    def _refresh_scope_limits(self, *_):
        """Bring the per-channel scope bounds up to date, computing missing ones off the GUI thread.

        Bounds are cached per channel, scope group, percentiles, axes and the
        group's loaded images, so a change to any of them is a new entry.
        """
        request = self._scope_request()
        if request is None:
            if self._channel_limits:
                self._set_display_layers(channel_limits={})
            return
        scoped, percentiles, axes, own = request
        axis_key = tuple(sorted(axes.items()))
        members = {}
        for field in self._fields:
            if field.unresolved or not field.well:
                continue
            for channel, scope in scoped.items():
                members.setdefault((channel, scope, _overview_scope_group(field, scope)), []).append(field)
        keys = {group: group + (own.get(group[0], percentiles), axis_key,
                                frozenset(field.identity for field in fields))
                for group, fields in members.items()}
        missing = {keys[group]: (group[0], tuple(fields)) for group, fields in members.items()
                   if keys[group] not in self._scope_cache}
        self._scope_generation += 1
        if not missing:
            self._install_scope_limits(scoped, keys)
            return
        self._metadata_jobs.submit(
            functools.partial(_overview_scope_statistics, missing, percentiles, axes, self._cancelled, own),
            functools.partial(self._scope_statistics_ready, self._scope_generation, scoped, keys))

    def _scope_statistics_ready(self, generation, scoped, keys, result):
        """Cache computed bounds and apply them while the request is still current."""
        self._scope_cache.update(result)
        if generation == self._scope_generation:
            self._install_scope_limits(scoped, keys)

    def _install_scope_limits(self, scoped, keys):
        """Give every loaded image its channels' scope bounds; redraw only when they changed."""
        limits = {}
        for field in self._fields:
            if field.unresolved or not field.well:
                continue
            entry = {}
            for channel, scope in scoped.items():
                bounds = self._scope_cache.get(keys.get((channel, scope, _overview_scope_group(field, scope))))
                if bounds is not None and bounds[1] > bounds[0]:
                    entry[channel] = bounds
            if entry:
                limits[_overview_image_key(field)] = entry
        if limits != self._channel_limits:
            self._set_display_layers(channel_limits=limits)

    def _set_display_layers(self, channel_limits=None, image_overrides=None):
        """Replace the scope limits or the per-image overrides and redraw every view.

        Each mapping is replaced, never changed in place, because workers
        read the one they were handed; a new mapping is also a new cache key.

        :param channel_limits: ``{image key: {channel: (low, high)}}``, or
            ``None`` to keep the current one.
        :param image_overrides: ``{image key: override}``, or ``None`` to keep it.
        """
        if channel_limits is not None:
            self._channel_limits = dict(channel_limits)
        if image_overrides is not None:
            self._image_overrides = dict(image_overrides)
        self._display_generation += 1
        self._thumbnail_cache.clear()
        self._displayed_montage = None
        self._displayed_field = None
        self._request_visuals()
        self._reload_image_views()
        item = self.fields.currentItem()
        if item is not None and self._individual:
            self._field(item, preserve_view=True)

    def _field_shown_in(self, widget, position=None):
        """The field a view shows: the enlarged image, a stacked image, or a well view's tile.

        :param widget: the image view, a stacked image's view, or a well's field list.
        :param position: for a field list, the point in its viewport.
        :returns: the :class:`_OverviewField`, or ``None``.
        """
        if widget is self.view:
            return self._main_view_field
        panel = next((panel for panel in self._image_views if panel.view is widget), None)
        if panel is not None:
            return panel.field
        if widget in self._field_lists() and position is not None:
            item = widget.itemAt(position)
            return item.data(Qt.UserRole) if item is not None else None
        return None

    def _persist_view_state(self):
        """Save every plate's collapsed lines and the per-image overrides to ``overview_view.json``."""
        if not self._metadata_source:
            return
        state = dict(self._view_state)
        plates = dict(state.get("plates") or {})
        for plate, (rows, columns) in self._collapsed.items():
            entry = dict(plates.get(_overview_key_text(self._plate_key(plate))) or {})
            entry.update(collapsed_rows=sorted(rows), collapsed_columns=sorted(columns))
            plates[_overview_key_text(self._plate_key(plate))] = entry
        state["plates"] = plates
        state["images"] = {_overview_key_text(key): value for key, value in self._image_overrides.items()}
        state["sources"] = list(getattr(self, "_source_paths", ()) or [self._metadata_source])
        state["options"] = {"use_subfolders": list(self._overview_options.get("use_subfolders") or ()),
                            "excluded_plates": list(self._overview_options.get("excluded_plates") or ()),
                            "max_wells_per_plate": self._load_limits[0],
                            "max_fields_per_well": self._load_limits[1],
                            "plate_regex": {str(key): str(value) for key, value in
                                            dict(self._overview_options.get("plate_regex") or {}).items() if value}}
        state["view"] = {"plates_view": self._plates_view,
                         "shown_plates": [_overview_key_text(self._plate_key(plate))
                                          for plate in self._plates_subset or ()]}
        self._view_state = state
        self._view_state_pending = (self._metadata_source, state)
        if not self._view_state_writing:
            self._write_view_state()

    def _write_view_state(self):
        """Write the latest view state; one write at a time, so the newest always lands last."""
        pending, self._view_state_pending = self._view_state_pending, None
        if pending is None:
            return
        self._view_state_writing = True
        self._metadata_jobs.submit(functools.partial(_overview_try_save_view_state, *pending),
                                   self._view_state_written)

    def _view_state_written(self, _result):
        """Start the next view-state write when one was asked for meanwhile."""
        self._view_state_writing = False
        self._write_view_state()

    def _restore_view_state(self):
        """Read ``overview_view.json`` off the GUI thread and apply it to this source's plates."""
        if not self._metadata_source:
            return
        source, fields, cancelled = self._metadata_source, self._fields, self._cancelled
        self._metadata_jobs.submit(functools.partial(_overview_load_view_state, source),
                                   functools.partial(self._view_state_loaded, fields, cancelled))

    def _view_state_loaded(self, fields, cancelled, state):
        """Install a read view state while the same source generation is shown."""
        import json

        if cancelled.is_set() or fields is not self._fields:
            return
        self._view_state = state
        restored = False
        saved = state.get("plates") or {}
        if saved:
            plates = {_overview_key_text(self._plate_key(plate)): plate for plate in {field.plate for field in fields}}
            for text, entry in saved.items():
                plate = plates.get(text)
                if plate is None or not isinstance(entry, dict):
                    continue
                lines = ({int(value) for value in entry.get("collapsed_rows") or ()},
                         {int(value) for value in entry.get("collapsed_columns") or ()})
                restored = restored or any(lines)
                self._collapsed[plate] = lines
        overrides = {}
        for text, value in (state.get("images") or {}).items():
            try:
                overrides[tuple(json.loads(text))] = value
            except (TypeError, ValueError):
                continue
        if overrides:
            self._set_display_layers(image_overrides=overrides)
        view = state.get("view") or {}
        if view.get("plates_view") in {value for value, _caption in _OVERVIEW_PLATES_VIEWS}:
            plates = {_overview_key_text(self._plate_key(plate)): plate for plate in {field.plate for field in fields}}
            subset = [plates[text] for text in view.get("shown_plates") or () if text in plates]
            restored = not self._set_plates_view(view["plates_view"], subset, persist=False) and restored
        if restored and self._plate_cell_extent is not None:
            self._relayout_collapsed()

    def _sync_plate_buttons(self):
        """Offer stepping between, removing and showing several plates only while several are loaded."""
        several = self.plates.count() > 1
        previous, following, _add, remove = self._plate_buttons
        for button in (previous, following, remove, *self._plates_view_controls):
            button.setEnabled(several)

    def _new_plate_table(self):
        """Create one plate's well table in its own block, under a header naming the plate.

        Every table shares the browser's selection, menus, collapse, hover
        readout and lazy thumbnails; its ``_overview_plate`` names the plate
        it shows.

        :returns: the table.
        """
        from PySide6.QtCore import QSize
        from PySide6.QtWidgets import QAbstractItemView, QTableWidget

        table = QTableWidget()
        table._overview_plate = None
        table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        table.horizontalHeader().setMinimumSectionSize(1)
        table.verticalHeader().setMinimumSectionSize(1)
        table.setItemDelegate(_OverviewPlateDelegate(table))
        table.setMouseTracking(True)
        table.viewport().setMouseTracking(True)
        table.setContextMenuPolicy(Qt.CustomContextMenu)
        table.customContextMenuRequested.connect(functools.partial(self._plate_table_menu, table))
        for axis, header in (("row", table.verticalHeader()), ("column", table.horizontalHeader())):
            header.setContextMenuPolicy(Qt.CustomContextMenu)
            header.customContextMenuRequested.connect(functools.partial(self._plate_header_menu, table, axis))
            header.sectionDoubleClicked.connect(functools.partial(self._plate_toggle_line, table, axis))
            header.viewport()._overview_header = (table, axis)
            header.viewport().installEventFilter(self)
        table.viewport().installEventFilter(self)
        table.installEventFilter(self)
        table.horizontalScrollBar().valueChanged.connect(self._request_visuals)
        table.verticalScrollBar().valueChanged.connect(self._request_visuals)
        table.setIconSize(QSize(160, 120))
        follow_device_ratio(table.viewport(), self._request_visuals)
        table.itemDoubleClicked.connect(self._well)
        table.itemClicked.connect(self._clicked_well)
        table.itemSelectionChanged.connect(self._remember_wells)
        table.setToolTip(tr("Select wells for export; double-click a well to explore its fields."))
        self._track_readout(table.viewport(), table)
        block = QWidget()
        block.setObjectName("FieldOverviewPlateBlock")
        layout = QVBoxLayout(block)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        header = QLabel()
        header.setObjectName("FieldOverviewPlateHeader")
        header.hide()
        layout.addWidget(header)
        layout.addWidget(table, 1)
        self._plate_blocks[table] = (block, header)
        self._plates_layout.addWidget(block, 1)
        return table

    def _plate_table_list(self):
        """The shown plates' tables, in plate order; :attr:`wells` alone before a plate is shown."""
        tables = tuple(getattr(self, "_plate_tables", {}).values())
        if tables or getattr(self, "wells", None) is None:
            return tables
        return (self.wells,)

    def _shown_plates(self):
        """The plates the plate level shows: the current one, or every chosen plate at once.

        :returns: session plate names in plate order; Side by side and
            Stacked show the chosen subset, or every loaded plate.
        """
        loaded = [self.plates.itemData(index) for index in range(self.plates.count())]
        current = self.plates.currentData()
        if self._plates_view == "single" or not loaded:
            return (current,) if current is not None else ()
        chosen = [plate for plate in loaded if self._plates_subset is None or plate in self._plates_subset]
        return tuple(chosen or loaded)

    def _sync_plate_tables(self):
        """Give every shown plate a table, in plate order, and hide the rest.

        The current plate's table is :attr:`wells`; in Single mode it is
        always the same table, as before. A current plate outside the shown
        ones gives way to the first shown plate.
        """
        shown = self._shown_plates()
        current = self.plates.currentData()
        if shown and current not in shown:
            current = shown[0]
            self.plates.blockSignals(True)
            self.plates.setCurrentIndex(self.plates.findData(current))
            self.plates.blockSignals(False)
        previous = self._plate_tables
        tables = {plate: previous[plate] for plate in shown if plate in previous}
        free = [table for table in self._plate_blocks
                if table not in tables.values() and (shown or table is not self.wells)]
        for plate in sorted(shown, key=lambda name: name != current):
            if plate in tables:
                continue
            table = self.wells if self.wells in free else (free[0] if free else self._new_plate_table())
            if table in free:
                free.remove(table)
            tables[plate] = table
        self._plate_tables = {plate: tables[plate] for plate in shown}
        for table in free:
            self._clear_plate_table(table)
            table._overview_plate = None
            self._plate_extents.pop(table, None)
            self._plate_blocks[table][0].hide()
        self._plates_layout.setDirection(QBoxLayout.LeftToRight if self._plates_view == "side"
                                         else QBoxLayout.TopToBottom)
        for index, table in enumerate(self._plate_tables.values()):
            block = self._plate_blocks[table][0]
            self._plates_layout.removeWidget(block)
            self._plates_layout.insertWidget(index, block, 1)
            block.show()
        if current in self._plate_tables:
            self.wells = self._plate_tables[current]
            self._well_viewport = self.wells.viewport()
        self._label_plate_blocks()

    def _clear_plate_table(self, table):
        """Empty one plate table without signalling a selection change."""
        table.blockSignals(True)
        table.clear()
        table.setRowCount(0)
        table.setColumnCount(0)
        table.blockSignals(False)

    def _label_plate_blocks(self):
        """Name each shown plate above its table; the current plate's name is bold.

        Single mode shows no plate header, as before.
        """
        names = {self.plates.itemData(index): self.plates.itemText(index) for index in range(self.plates.count())}
        several = self._plates_view != "single"
        for table, (_block, header) in self._plate_blocks.items():
            plate = getattr(table, "_overview_plate", None)
            header.setVisible(several and plate is not None and table in self._plate_tables.values())
            if plate is None:
                continue
            header.setText(tr("Plate {plate}", plate=names.get(plate, plate)))
            font = header.font()
            font.setBold(table is self.wells)
            header.setFont(font)

    def _focus_plate_table(self, table):
        """Make the plate ``table`` shows the current plate, without rebuilding any table.

        Its menus, collapse and well opening then act on that plate. Wells
        open from another plate close, as when another plate is chosen.

        :param table: a shown plate table.
        :returns: whether the current plate changed.
        """
        plate = getattr(table, "_overview_plate", None)
        if table is self.wells or plate is None or self._plate_tables.get(plate) is not table:
            return False
        self.wells, self._well_viewport = table, table.viewport()
        self.plates.blockSignals(True)
        self.plates.setCurrentIndex(self.plates.findData(plate))
        self.plates.blockSignals(False)
        if self._scope[0] != plate:
            for panel in tuple(self._well_panels):
                self._close_well_panel(panel, refresh=False)
            self.fields = self._well_panel.field_list
            self._scope = (plate, None)
            self._visible = tuple(self._plate_available.get(plate, ()))
            self._page = 0
        self._plate_cell_extent = self._plate_extents.get(table, self._plate_cell_extent)
        self._label_plate_blocks()
        return True

    def _plate_table_menu(self, table, position):
        """Make ``table``'s plate current, then offer :meth:`_well_menu`."""
        self._focus_plate_table(table)
        return self._well_menu(position)

    def _plate_header_menu(self, table, axis, position):
        """Make ``table``'s plate current, then offer :meth:`_header_menu`."""
        self._focus_plate_table(table)
        return self._header_menu(axis, position)

    def _plate_toggle_line(self, table, axis, index):
        """Make ``table``'s plate current, then collapse or show one of its lines."""
        self._focus_plate_table(table)
        self._toggle_line(axis, index)

    def _plates_view_chosen(self, *_):
        """Follow the Plates view choice; see :meth:`_set_plates_view`."""
        self._set_plates_view(self.plates_view.currentData())

    def _set_plates_view(self, mode=None, plates=None, persist=True):
        """Show one plate, or several plates at once side by side or stacked.

        :param mode: ``"single"``, ``"side"`` or ``"stacked"``; ``None`` keeps the mode.
        :param plates: session names of the plates Side by side and Stacked
            show, ``()`` for every plate, or ``None`` to keep the choice.
        :param persist: save the choice to ``overview_view.json``.
        :returns: whether the view changed.
        """
        modes = tuple(value for value, _caption in _OVERVIEW_PLATES_VIEWS)
        mode = self._plates_view if mode not in modes else mode
        before = (self._plates_view, self._plates_subset)
        if plates is not None:
            self._plates_subset = tuple(dict.fromkeys(str(plate) for plate in plates)) or None
        self._plates_view = mode
        changed = (self._plates_view, self._plates_subset) != before
        index = self.plates_view.findData(mode)
        if index != self.plates_view.currentIndex():
            self.plates_view.blockSignals(True)
            self.plates_view.setCurrentIndex(index)
            self.plates_view.blockSignals(False)
        if not changed:
            return False
        self._plate_fit = True
        self._plate_cell_extent = None
        if self._fields:
            self._filter()
        else:
            self._sync_plate_tables()
        if persist:
            self._persist_view_state()
        return True

    def _fill_shown_plates_menu(self):
        """List every loaded plate as a tick in the Shown plates menu, with All plates first."""
        menu = self._shown_plates_menu
        menu.clear()
        every = menu.addAction(tr("All plates"))
        every.setCheckable(True)
        every.setChecked(self._plates_subset is None)
        every.triggered.connect(self._show_every_plate)
        menu.addSeparator()
        for index in range(self.plates.count()):
            plate = self.plates.itemData(index)
            action = menu.addAction(self.plates.itemText(index))
            action.setCheckable(True)
            action.setChecked(self._plates_subset is None or plate in self._plates_subset)
            action.toggled.connect(functools.partial(self._toggle_shown_plate, plate))
        return menu

    def _show_every_plate(self, *_):
        """Show every loaded plate in Side by side and Stacked."""
        self._set_plates_view(None, ())

    def _toggle_shown_plate(self, plate, shown):
        """Add ``plate`` to the shown plates or take it out; at least one plate stays shown."""
        loaded = [self.plates.itemData(index) for index in range(self.plates.count())]
        chosen = [name for name in loaded if self._plates_subset is None or name in self._plates_subset]
        if shown and plate not in chosen:
            chosen.append(plate)
        elif not shown and plate in chosen and len(chosen) > 1:
            chosen.remove(plate)
        chosen = [name for name in loaded if name in chosen]
        self._set_plates_view(None, () if len(chosen) == len(loaded) else chosen)

    def _step_plate(self, delta, *_):
        """Show the next (``delta`` 1) or previous (-1) loaded plate, wrapping around.

        Side by side and Stacked step between the shown plates.
        """
        count = self.plates.count()
        shown = self._shown_plates()
        if self._plates_view != "single" and len(shown) > 1:
            index = shown.index(self.plates.currentData()) if self.plates.currentData() in shown else 0
            self.plates.setCurrentIndex(self.plates.findData(shown[(index + int(delta)) % len(shown)]))
        elif count > 1:
            self.plates.setCurrentIndex((self.plates.currentIndex() + int(delta)) % count)

    def _choose_plates(self, *_):
        """Pick one or more plate folders, or folders of plates, and add them.

        :returns: the chosen paths.
        """
        dialog = _overview_folders_dialog(self, tr("Add plates…"))
        self._add_plates_dialog = dialog
        paths = dialog.selectedFiles() if dialog.exec() else []
        if paths:
            self._add_sources(paths)
        return paths

    def _add_sources(self, paths):
        """Index more plate folders, or folders of plates, and add their plates.

        Indexing runs on a job thread; a plate whose name is taken gets a
        number after it.

        :param paths: the folders to add.
        """
        known = set(_overview_source_paths(self.source.text()))
        paths = [path for path in _overview_source_paths(paths) if path not in known]
        if not paths:
            return
        regex, convention = self.regex.text(), self._convention
        scope = self._overview_options.get("source_scope", "recursive")
        layout = self._overview_options.get("well_layout")
        use_subfolders = tuple(self._overview_options.get("use_subfolders") or ())
        loaded = {field.source for field in self._fields}
        excluded = tuple(self._overview_options.get("excluded_plates") or ()) + tuple(loaded)
        self.status.setText(tr("Indexing image metadata…"))
        patterns = dict(self._overview_options.get("plate_regex") or {})
        self._index_jobs.submit(functools.partial(
            _overview_index_sources, paths, regex, convention, self._cancelled, scope, layout,
            use_subfolders, excluded, True, patterns), functools.partial(self._sources_added, paths, self._fields))

    def _sources_added(self, paths, fields, result):
        """Merge newly indexed plates into the Overview, keeping every plate name unique."""
        import dataclasses

        if fields is not self._fields or self._cancelled.is_set():
            return
        added, _detection = result
        added = _overview_matched_fields(added, self.regex.text(), self._overview_options.get("plate_regex"))
        taken = {field.plate for field in self._all_fields}
        renamed = {}
        for plate in sorted({field.plate for field in added}, key=_overview_natural_key):
            name, number = plate, 2
            while name in taken:
                name, number = f"{plate} ({number})", number + 1
            taken.add(name)
            renamed[plate] = name
        added = tuple(dataclasses.replace(field, plate=renamed[field.plate]) for field in added)
        self.source.setText("; ".join(_overview_source_paths(self.source.text()) + list(paths)))
        self._indexed(tuple(self._all_fields) + added)
        self._persist_view_state()

    def _remove_shown_plate(self, *_):
        """Remove the shown plate; see :meth:`_remove_plate`."""
        if self.plates.count() > 1:
            self._remove_plate(self.plates.currentData())

    def _remove_plate(self, plate):
        """Take one plate out of the Overview and drop everything held for it.

        Its fields, selections, collapsed lines and cached previews go; its
        saved annotations and measurements stay on disk. A plate that came
        from a folder of plates is remembered as removed, so a reload does
        not bring it back.

        :param plate: the plate's session name.
        """
        removed = tuple(field for field in self._all_fields if field.plate == plate)
        if not removed or len(removed) == len(self._all_fields):
            return
        identities = {field.identity for field in removed}
        self._drop_cached(identities)
        for address in [address for address in self._selected_wells if address[0] == plate]:
            self._selected_wells.discard(address)
            self._selected_well_axes.pop(address, None)
            self._selected_well_options.pop(address, None)
        for identity in [identity for identity in self._selected_fields if identity in identities]:
            self._selected_fields.pop(identity, None)
            self._selected_field_options.pop(identity, None)
        self._collapsed.pop(plate, None)
        if self._plates_subset is not None:
            self._plates_subset = tuple(name for name in self._plates_subset if name != plate) or None
        folders = {os.path.realpath(field.source) for field in removed if field.source}
        given = _overview_source_paths(self.source.text())
        kept = [path for path in given if os.path.realpath(path) not in folders]
        excluded = [path for path in folders if path not in {os.path.realpath(path) for path in given}]
        options = dict(self._overview_options)
        options["excluded_plates"] = tuple(dict.fromkeys(
            tuple(options.get("excluded_plates") or ()) + tuple(excluded)))
        self._overview_options = options
        if kept != given:
            self.source.setText("; ".join(kept))
        self._indexed(tuple(field for field in self._all_fields if field.plate != plate))
        self._persist_view_state()

    def _drop_cached(self, identities):
        """Forget the cached previews of the fields with these identities."""
        for cache in (self._cache, self._thumbnail_cache):
            for key in list(cache):
                head = key[0] if isinstance(key, tuple) and key else key
                if head in identities or (isinstance(head, tuple) and any(part in identities for part in head)):
                    cache.pop(key, None)

    def _set_load_limits(self, max_wells=None, max_fields=None):
        """Load at most ``max_wells`` wells per plate and ``max_fields`` fields per well.

        Only the difference is reloaded: raising a cap adds the wells or
        fields it lets in, lowering one drops theirs from the caches; the
        indexed source is not read again.

        :param max_wells: the Max wells per plate setting.
        :param max_fields: the Max fields per well setting.
        """
        limits = (_overview_load_limit(max_wells), _overview_load_limit(max_fields))
        if limits == self._load_limits:
            return
        self._load_limits = limits
        if not self._all_fields:
            return
        kept = {field.identity for field in _overview_limited_fields(self._all_fields, *limits)}
        self._drop_cached({field.identity for field in self._fields if field.identity not in kept})
        for identity in [identity for identity in self._selected_fields if identity not in kept]:
            self._selected_fields.pop(identity, None)
            self._selected_field_options.pop(identity, None)
        self._indexed(self._all_fields)

    def _plate_key(self, plate=None):
        """The stable key (:func:`_overview_plate_key`) of a plate shown in this session.

        :param plate: the plate's session name; the shown plate when ``None``.
        """
        plate = self.plates.currentData() if plate is None else plate
        field = next((field for field in self._fields if field.plate == plate), None)
        return _overview_plate_key(field) if field is not None else ("", str(plate or ""))

    def _set_enhancement(self, chain):
        """Show every view through Make Masks' enhancement chain, or without one.

        :param chain: the chain, or ``None`` for no enhancement.
        """
        if chain == self._enhancement:
            return
        self._enhancement = chain
        self._cache.clear()
        self._thumbnail_cache.clear()
        self._displayed_montage = None
        self._displayed_field = None
        self._request_visuals()
        self._reload_image_views()
        item = self.fields.currentItem()
        if item is not None and self._individual:
            self._field(item, preserve_view=True)

    def _new_well_panel(self, fields):
        """Wrap one field list in a level container with metadata and close actions.

        :param fields: the well's field thumbnail list.
        :returns: the hidden, unopened well container.
        """
        from .ai_toggle_label import AiToggleLabel
        from ..theme import apply_close_mark

        from PySide6.QtWidgets import QSlider, QSpinBox

        panel = _OverviewPanel("")
        panel.field_list = fields
        panel.body_layout.addWidget(fields)
        panel.tile_slider = QSlider(Qt.Horizontal)
        panel.tile_slider.setObjectName("FieldOverviewTileSize")
        panel.tile_slider.setRange(*_OVERVIEW_TILE_RANGE)
        panel.tile_slider.setValue(self._tile_size)
        panel.tile_slider.setToolTip(tr("Size of the images in this well view"))
        panel.tile_slider.setAccessibleName(tr("Image size"))
        panel.tile_slider.setMaximumWidth(160)
        panel.tile_slider.valueChanged.connect(self._set_tile_size)
        self._tile_sliders.append(panel.tile_slider)
        panel.tile_spacing = QSpinBox()
        panel.tile_spacing.setObjectName("FieldOverviewTileSpacing")
        panel.tile_spacing.setRange(0, _OVERVIEW_TILE_SPACING_MAX)
        panel.tile_spacing.setValue(self._tile_spacing)
        panel.tile_spacing.setSuffix(tr(" px"))
        panel.tile_spacing.setToolTip(tr("Overview tile spacing: the gap, in screen pixels, between the images "
                                         "of every well view."))
        panel.tile_spacing.setAccessibleName(tr("Overview tile spacing"))
        panel.tile_spacing.valueChanged.connect(self._set_tile_spacing)
        self._tile_spacing_boxes.append(panel.tile_spacing)
        panel.tile_spacing.setMaximumWidth(64)
        panel.header_layout.addWidget(panel.tile_spacing)
        panel.header_layout.addWidget(panel.tile_slider)
        panel.metadata_toggle = AiToggleLabel(text="metadata", tooltip=_OVERVIEW_METADATA_TOOLTIP)
        panel.metadata_toggle.toggled.connect(
            lambda on, owner=fields: self._well_metadata_toggled(owner, on))
        panel.header_layout.addWidget(panel.metadata_toggle)
        panel.close_button = QToolButton()
        apply_close_mark(panel.close_button, tooltip=tr("Close this well view"))
        panel.close_button.setAccessibleName(tr("Close this well view"))
        panel.close_button.clicked.connect(lambda _checked=False, owner=panel: self._close_well_panel(owner))
        panel.header_layout.addWidget(panel.close_button)
        panel.hide()
        return panel

    def _field_lists(self):
        """Return the active field list followed by every other open well view.

        :returns: distinct lists in display priority.
        """
        lists = [self.fields]
        lists.extend(panel.field_list for panel in self._well_panels
                     if panel.field_list is not self.fields)
        return lists

    def _open_well_panel(self, well):
        """Open or reuse one stacked well container and make it active.

        :param well: the physical well identity.
        """
        panel = next((panel for panel in self._well_panels
                      if panel.field_list._overview_well == well), None)
        if panel is None:
            if len(self._well_panels) >= _OVERVIEW_MAX_WELL_VIEWS:
                self._close_well_panel(self._well_panels[0], refresh=False)
            panel = (self._well_panel if self._well_panel not in self._well_panels
                     else self._new_well_panel(self._new_field_list()))
            panel.field_list._overview_well, panel.field_list._overview_page = well, 0
            panel.title.setText(tr("Well {well}", well=well or tr("Unresolved")))
            self._well_panels.append(panel)
            self._levels.addWidget(panel)
            panel.set_collapsed(False)
            panel.show()
        self.fields = panel.field_list
        self._stack_open_wells()

    def _stack_open_wells(self):
        """With two or more wells open, stack one image from each side by side."""
        if len(self._well_panels) < 2 or not self._normalization_ready:
            return
        fields = []
        for panel in self._well_panels:
            address = (self._scope[0], panel.field_list._overview_well)
            candidates = getattr(self, "_well_fields", {}).get(address) or ()
            representative = self._well_representatives.get(address)
            chosen = representative if representative is not None else (candidates[0] if candidates else None)
            if chosen is not None:
                fields.append(chosen)
        shown = {(panel.field.plate, panel.field.well) for panel in self._image_views}
        fresh = [field for field in fields if (field.plate, field.well) not in shown]
        if fresh:
            self._open_image_views(fresh)

    def _close_well_panel(self, panel, refresh=True):
        """Close one stacked well view, keeping the other open wells unchanged.

        :param panel: the well container to close.
        :param refresh: refresh navigation after closing the active view.
        """
        self._well_panels.remove(panel)
        panel.hide()
        if panel is not self._well_panel:
            for controls, control in ((self._tile_sliders, panel.tile_slider),
                                      (self._tile_spacing_boxes, panel.tile_spacing)):
                if control in controls:
                    controls.remove(control)
            panel.setParent(None)
            panel.deleteLater()
        if panel.field_list is not self.fields or not refresh:
            return
        if not self._well_panels:
            self._back()
            return
        self._activate_well_list(self._well_panels[-1].field_list)
        self._filter()

    def _activate_well_list(self, fields):
        """Make another open well view the navigation target without rebuilding it.

        :param fields: the open well's field list.
        """
        if fields is self.fields:
            return
        self.fields = fields
        self._scope = (self.plates.currentData(), fields._overview_well)
        self._page = fields._overview_page
        self._visible = tuple(self._well_fields.get((self._scope[0], fields._overview_well), ()))
        self._individual = False
        self._comparison_signature = None
        self._displayed_montage = None

    def _field_in(self, fields, item):
        """Enlarge a clicked field from any open well view.

        :param fields: the list that received the click.
        :param item: the clicked field item.
        """
        self._activate_well_list(fields)
        self._field(item)

    def _current_field_in(self, fields, current, previous):
        """Follow keyboard navigation in whichever well view has focus.

        :param fields: the list whose current item changed.
        :param current: the new current item, or ``None`` during a reset.
        :param previous: the earlier current item.
        """
        if current is not None:
            self._activate_well_list(fields)
        self._current_field_changed(current, previous)

    def _fill_field_list(self, fields, records, current_identity):
        """Show one well view's bounded field page without decoding pixels.

        :param fields: the list to fill.
        :param records: the field records on its current page.
        :param current_identity: the identity to keep current, if present.
        """
        from PySide6.QtWidgets import QListWidgetItem
        from PySide6.QtCore import QItemSelectionModel

        fields.blockSignals(True)
        fields.clear()
        for field in records:
            caption = tr("{well} · field {field} · T {time} · Z {z}",
                well=field.well or tr("Unresolved"), field=getattr(field, "field_id", "") or field.field,
                time=getattr(field, "time_id", "") or str(self.time_axis.currentData()),
                z=getattr(field, "z_id", "") or str(self.z_axis.currentData()))
            item = QListWidgetItem(caption if fields._overview_metadata else "")
            item.setData(Qt.UserRole + 2, caption)
            item.setToolTip(caption + "\n" + field.caption)
            item.setData(Qt.UserRole, field)
            fields.addItem(item)
            item.setSelected(field.identity in self._selected_fields)
            if field.identity == current_identity:
                fields.setCurrentItem(item, QItemSelectionModel.NoUpdate)
        fields.blockSignals(False)

    def _well_metadata_toggled(self, fields, on):
        """Show or hide the captions of one well view's thumbnails.

        :param fields: the well view's list.
        :param on: whether the caption text is visible.
        """
        fields._overview_metadata = bool(on)
        self._apply_tile_geometry(fields)
        for index in range(fields.count()):
            item = fields.item(index)
            item.setText(item.data(Qt.UserRole + 2) if on else "")

    def _image_metadata_toggled(self, on):
        """Show or hide filename and metadata text over the enlarged image.

        :param on: whether the text is visible.
        """
        self._update_image_metadata()
        self._request_visuals()

    def _update_image_metadata(self, field=None):
        """Place the current field's filename and metadata over the image view.

        :param field: the enlarged field, or ``None`` to keep the current text.
        """
        if field is not None:
            names = "\n".join(Path(path).name for _channel, path in field.files)
            pixel = self._overview_options.get("pixel_size_um")
            pixel = "\n" + tr("Pixel size: {size} µm", size=f"{pixel:g}") if pixel else ""
            self._image_metadata.setText(field.caption + ("\n" + names if names else "") + pixel)
        shown = self.metadata_toggle.isChecked() and self._individual
        self._image_metadata.setVisible(shown)
        if shown:
            self._image_metadata.setMaximumWidth(max(80, self.view.viewport().width() - 16))
            self._image_metadata.adjustSize()
            self._image_metadata.move(8, 8)
            self._image_metadata.raise_()

    def _show_image_column(self):
        """Reveal the image column to the right of the plate and well levels."""
        if self._image_column.isHidden():
            self._image_column.show()
            total = max(2, sum(self._splitter.sizes()))
            self._splitter.setSizes([total * 2 // 5, total - total * 2 // 5])

    def _show_image_panel(self):
        """Reveal the image level to the right of the plate and well levels."""
        self._image_panel.show()
        self._show_image_column()

    def _open_selected_images(self):
        """Open one stacked enlarged image per selected field, across wells."""
        fields = tuple(self._selected_fields.values())
        if not fields:
            self.status.setText(tr("Select fields by clicking or dragging a selection rectangle."))
            return
        self._open_image_views(fields)

    def _open_image_views(self, fields):
        """Stack enlarged images, closing the oldest beyond the maximum.

        :param fields: the fields to enlarge, in display order.
        """
        fields = tuple(fields)[-_OVERVIEW_MAX_IMAGE_VIEWS:]
        for field in fields:
            if any(panel.field.identity == field.identity for panel in self._image_views):
                continue
            if len(self._image_views) >= _OVERVIEW_MAX_IMAGE_VIEWS:
                self._close_image_view(self._image_views[0])
            self._image_views.append(self._new_image_view(field))
        self._show_image_column()

    def _new_image_view(self, field):
        """Build one closable, labelled enlarged image and start loading it.

        :param field: the field it shows.
        :returns: the image container, already in the stack.
        """
        from .ai_toggle_label import AiToggleLabel
        from ..theme import apply_close_mark

        panel = _OverviewPanel(tr("Well {well}, field {field}", well=field.well or tr("Unresolved"),
                                  field=field.field))
        panel.setProperty("overviewStackedImage", True)
        panel.field = field
        panel.generation = 0
        panel.view = _OverviewImageView()
        panel.view.setMinimumHeight(self._image_view_height())
        panel.body_layout.addWidget(panel.view)
        panel.view.detail_requested.connect(lambda owner=panel.view: self._sync_image_views(owner))
        self._track_readout(panel.view.viewport(), panel.view)
        panel.view.setContextMenuPolicy(Qt.CustomContextMenu)
        panel.view.customContextMenuRequested.connect(functools.partial(self._image_menu, panel.view))
        panel.adjusted_badge = self._adjusted_badge()
        panel.adjusted_badge.setVisible(self._is_adjusted(field))
        panel.header_layout.addWidget(panel.adjusted_badge)
        for bar in (panel.view.horizontalScrollBar(), panel.view.verticalScrollBar()):
            bar.valueChanged.connect(lambda _value, owner=panel.view: self._sync_image_views(owner))
        panel.metadata_label = QLabel(panel.view.viewport())
        panel.metadata_label.setObjectName("FieldOverviewImageMetadata")
        panel.metadata_label.setWordWrap(True)
        names = "\n".join(Path(path).name for _channel, path in field.files)
        panel.metadata_label.setText(field.caption + ("\n" + names if names else ""))
        panel.metadata_label.hide()
        panel.metadata_toggle = AiToggleLabel(text="metadata", tooltip=_OVERVIEW_METADATA_TOOLTIP)
        panel.metadata_toggle.toggled.connect(lambda on, owner=panel: self._image_view_metadata(owner, on))
        panel.close_button = QToolButton()
        apply_close_mark(panel.close_button, tooltip=tr("Close this image"))
        panel.close_button.setAccessibleName(tr("Close this image"))
        panel.close_button.clicked.connect(lambda _checked=False, owner=panel: self._close_image_view(owner))
        for widget in (panel.metadata_toggle, panel.close_button):
            panel.header_layout.addWidget(widget)
        self._image_stack.addWidget(panel)
        self._load_image_view(panel)
        return panel

    def _sync_image_views(self, source):
        """Give every stacked image the zoom and position of the one being moved, with Link zoom on.

        Off by default, so a wheel or pinch zooms only the image under the
        pointer. With Link zoom ticked in an image's right-click menu, the
        same region of every stacked image lines up for comparison by eye.

        :param source: the image view the user zoomed or scrolled.
        """
        if not self._link_zoom or getattr(self, "_syncing_views", False):
            return
        self._syncing_views = True
        try:
            for panel in self._image_views:
                view = panel.view
                if view is source:
                    continue
                view.setTransform(source.transform())
                view._user_zoomed = source._user_zoomed
                view.horizontalScrollBar().setValue(source.horizontalScrollBar().value())
                view.verticalScrollBar().setValue(source.verticalScrollBar().value())
        finally:
            self._syncing_views = False

    def _image_view_height(self):
        """The stacked images' height, following the image-size slider."""
        return max(96, round(self._tile_size * 1.5))

    def _image_view_metadata(self, panel, on):
        """Show or hide one stacked image's filename and metadata text.

        :param panel: the stacked image.
        :param on: whether the text is visible.
        """
        panel.metadata_label.setVisible(bool(on))
        if on:
            panel.metadata_label.setMaximumWidth(max(80, panel.view.viewport().width() - 16))
            panel.metadata_label.adjustSize()
            panel.metadata_label.move(8, 8)
            panel.metadata_label.raise_()

    def _load_image_view(self, panel):
        """Decode one stacked image at preview or original resolution.

        :param panel: the stacked image to (re)load.
        """
        try:
            channels, limits = self._choices()
        except (ValueError, TypeError) as exc:
            self.status.setText(str(exc))
            return
        panel.generation += 1
        generation, field = panel.generation, panel.field
        edge = None if self._quality_original() else self._quality_cap()
        options, cancelled = self._display_kwargs(view=True), self._cancelled
        plain_options = options
        heavy = self._original_chain(options) if edge is None else None
        if heavy is not None:
            options = dict(options, enhancement=heavy)
        def done(result):
            """Show the pixels only in the same, still open stacked image.

            :param result: detached pixels and an optional refusal.
            """
            pixels, error, note = result
            if cancelled.is_set() or panel not in self._image_views or panel.generation != generation:
                return
            if error:
                self.status.setText(tr("Could not read {field}: {error}", field=field.caption, error=error))
                return
            if heavy is not None:
                self.status.setText(note or tr("Original image shown with {steps}.",
                                               steps=_overview_heavy_steps(heavy)))
            image = QImage(pixels.data, pixels.shape[1], pixels.shape[0],
                           pixels.strides[0], QImage.Format_RGB888).copy()
            _overview_set_keeping_view(panel.view, QPixmap.fromImage(image))
        def read():
            """Read one detached image without Qt access."""
            try:
                return _overview_pixels(field, channels, limits, edge=edge, **options), "", ""
            except Exception as error:
                if heavy is None:
                    return None, str(error), ""
                note = tr("Original image shown without {steps}: {error}",
                          steps=_overview_heavy_steps(heavy), error=error)
                try:
                    return _overview_pixels(field, channels, limits, edge=edge, **plain_options), "", note
                except Exception as exc:
                    return None, str(exc), ""
            except Exception as exc:
                return None, str(exc), ""
        if heavy is not None:
            self.status.setText(tr("Applying {steps} to the original image…",
                                   steps=_overview_heavy_steps(heavy)))
        self._image_stack_jobs.submit(read, done)

    def _close_image_view(self, panel):
        """Close one stacked image, keeping every other open image.

        :param panel: the stacked image to close.
        """
        if panel not in self._image_views:
            return
        self._image_views.remove(panel)
        self._readout_viewports.pop(panel.view.viewport(), None)
        if self._readout_pending is not None and self._readout_pending[0] is panel.view:
            self._clear_pixel_readout()
        panel.hide()
        panel.setParent(None)
        panel.deleteLater()
        if not self._image_views and self._image_panel.isHidden():
            self._image_column.hide()

    def _reload_image_views(self):
        """Reload every stacked image after display choices change."""
        if self._controls_lost:
            return
        for panel in self._image_views:
            self._load_image_view(panel)

    def _well_menu(self, position):
        """Offer annotation of the selected wells on a right click.

        A right click on an unselected well selects only that well first.

        :param position: the click position in viewport coordinates.
        :returns: the shown context menu.
        """
        from PySide6.QtWidgets import QMenu

        item = self.wells.itemAt(position)
        if item is not None and item.data(_OVERVIEW_NOT_LOADED_ROLE):
            item = None
        if item is not None and not item.isSelected():
            for table in self._plate_table_list():
                table.clearSelection()
            item.setSelected(True)
        from ...schema import letters_from_row_index

        menu = QMenu(self.wells)
        action = menu.addAction(tr("Annotate selected wells…"))
        action.setEnabled(bool(self._selected_wells))
        action.triggered.connect(self._annotate_selected_wells)
        plate = self.plates.currentData()
        on_plate = tuple(field for field in self._fields if field.plate == plate and not field.unresolved)
        clicked = self._well_representatives.get((plate, item.data(Qt.UserRole))) if item is not None else None
        if clicked is not None and not clicked.unresolved:
            row = tuple(sorted({(plate, field.well) for field in on_plate if field.row == clicked.row}))
            column = tuple(sorted({(plate, field.well) for field in on_plate if field.column == clicked.column}))
            menu.addAction(tr("Annotate row {row}…", row=letters_from_row_index(int(clicked.row)))
                           ).triggered.connect(functools.partial(self._annotate_wells, row))
            menu.addAction(tr("Annotate column {column}…", column=clicked.column)
                           ).triggered.connect(functools.partial(self._annotate_wells, column))
        whole = tuple(sorted({(plate, field.well) for field in on_plate}))
        action = menu.addAction(tr("Annotate the whole plate…"))
        action.setEnabled(bool(whole))
        action.triggered.connect(functools.partial(self._annotate_wells, whole))
        menu.addSeparator()
        if item is not None:
            self._line_actions(menu, "row", item.row())
            self._line_actions(menu, "column", item.column())
        self._show_all_action(menu)
        menu.addSeparator()
        menu.addAction(tr("Import annotation scheme…")).triggered.connect(self._import_annotation_scheme)
        menu.addAction(tr("Save annotations…")).triggered.connect(self._save_annotations)
        menu.addAction(tr("Load annotations…")).triggered.connect(self._load_annotations)
        self._well_context_menu = menu
        menu.popup(self.wells.viewport().mapToGlobal(position))
        return menu

    def _annotate_selected_wells(self):
        """Open the key/value prompt for every selected plate-qualified well.

        :returns: the open prompt, or ``None`` without a selection.
        """
        return self._annotate_wells(tuple(sorted(self._selected_wells)))

    def _annotate_wells(self, wells, *_):
        """Open the key/value prompt for ``wells``.

        :param wells: plate-qualified ``(plate, well)`` identities.
        :returns: the open prompt, or ``None`` when there are no wells.
        """
        wells = tuple(wells)
        if not wells:
            self.status.setText(tr("Choose a row, column or well selection before annotating."))
            return None
        keys = sorted({key for values in self._annotation_snapshot().values() for key in values})
        dialog = _OverviewAnnotationDialog(keys, wells, self)
        dialog.accepted.connect(lambda: self._apply_annotation(set(dialog.wells), *dialog.annotation()))
        self._annotation_dialog = dialog
        dialog.open()
        return dialog

    def _build_comparison_controls(self):
        """Create the annotation-dimension picker and representative-image choices.

        :returns: the widget placed in the Comparisons settings category.
        """
        from PySide6.QtWidgets import QTreeWidget
        from .setup_dialog import _SetupComboBox

        host = QWidget(self)
        layout = QVBoxLayout(host)
        layout.setContentsMargins(0, 0, 0, 0)
        from .sortable_table import install_sorting

        self.comparison_dimensions = QTreeWidget()
        self.comparison_dimensions.setHeaderHidden(True)
        install_sorting(self.comparison_dimensions)
        self.comparison_dimensions.setMinimumHeight(120)
        self.comparison_dimensions.setToolTip(tr(
            "Tick annotation columns and the values to compare; every combination of the ticked values is one group."))
        self.comparison_within = _SetupComboBox()
        self.comparison_measurement = _SetupComboBox()
        self.comparison_measurement.setEditable(True)
        self.comparison_measurement.setToolTip(tr(
            "Image-level measurement to compare; leave empty to use the first measured column."))
        self.representative_method = _SetupComboBox()
        for text, value in ((tr("Closest to the class median"), "median"),
                            (tr("Class medoid"), "medoid"),
                            (tr("Closest to the class centroid in principal components"), "pca_centroid"),
                            (tr("Closest to the class median of the chosen measurement"), "measurement"),
                            (tr("Typical, in focus and unsaturated"), "typical_in_focus")):
            self.representative_method.addItem(text, value)
        self.group_builder = _OverviewGroupBuilder(
            lambda definition: _overview_group_count(self._fields, self._annotation_snapshot(), definition),
            lambda: self._comparison_selection()[1], self)
        self._restoring_groups = False
        self._groups_saving = self._groups_dirty = False
        self._group_jobs = JobRunner(self, app_key="field_overview", user_visible=False)
        self._group_jobs.job_finished.connect(self._comparison_groups_saved)
        self._group_jobs.job_failed.connect(self.status.setText)
        self.group_builder.changed.connect(self._save_comparison_groups)
        compare = QPushButton(tr("Compare annotated groups"))
        compare.clicked.connect(self._compare_groups)
        representatives = QPushButton(tr("Show representative images"))
        representatives.clicked.connect(self._show_representatives)
        self._comparison_buttons = (compare, representatives)
        for widget in (self.comparison_dimensions, QLabel(tr("Group values")), self.group_builder.picker_host,
                       QLabel(tr("Groups")), self.group_builder.group_host, self.group_builder.preset_host,
                       QLabel(tr("Compare within")), self.comparison_within,
                       QLabel(tr("Measurement")), self.comparison_measurement, compare,
                       QLabel(tr("Representative image")), self.representative_method, representatives):
            layout.addWidget(widget)
        self._refresh_comparison_choices()
        return host

    def _save_comparison_groups(self, *_):
        """Keep the built groups and their names in the source's measurements database.

        One save runs at a time; changes made meanwhile are saved, as they
        then stand, when it finishes, so the database ends with the last state.
        """
        if self._restoring_groups or not self._metadata_source:
            return
        if self._groups_saving:
            self._groups_dirty = True
            return
        self._groups_saving, self._groups_dirty = True, False
        self._group_jobs.submit(functools.partial(
            _overview_store_groups, self._metadata_source,
            self.group_builder.definitions(), self.group_builder.names()))

    def _comparison_groups_saved(self, *_):
        """Save again when the groups changed while the last save ran."""
        self._groups_saving = False
        if self._groups_dirty:
            self._save_comparison_groups()

    def _restore_comparison_groups(self):
        """Bring back the groups saved for this source, unless groups were built already."""
        if not self._metadata_source:
            return
        self._metadata_jobs.submit(functools.partial(_overview_saved_groups, self._metadata_source),
                                   functools.partial(self._comparison_groups_read, self._fields))

    def _comparison_groups_read(self, fields, saved):
        """Fill an empty group list with the saved groups and names.

        :param fields: the indexed fields the groups were read for.
        :param saved: ``(groups, names)``.
        """
        groups, names = saved
        builder = self.group_builder
        if fields is not self._fields or builder.groups.count() or not groups:
            return
        self._restoring_groups = True
        try:
            for definition, name in zip(groups, names):
                if builder.add_group(definition) and name != _overview_group_label(definition):
                    builder.rename_group(row=builder.groups.count() - 1, name=name)
        finally:
            self._restoring_groups = False

    def _refresh_comparison_choices(self):
        """Offer every annotation column and value, keeping earlier ticks."""
        from .sortable_table import tree_item

        tree = self.comparison_dimensions
        ticked = {(item.parent().text(0) if item.parent() else None, item.text(0))
                  for item in self._comparison_items() if item.checkState(0) == Qt.Checked}
        values = {}
        for metadata in self._annotation_snapshot().values():
            for key, value in metadata.items():
                values.setdefault(key, set()).add(value)
        plates = {field.plate for field in self._fields}
        if len(plates) > 1:
            values.setdefault("plate", set()).update(plates)
        tree.clear()
        for key in sorted(values):
            parent = tree_item([key])
            parent.setFlags(parent.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
            tree.addTopLevelItem(parent)
            for value in sorted(values[key], key=_overview_natural_key):
                child = tree_item([value])
                child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                child.setCheckState(0, Qt.Checked if (key, value) in ticked else Qt.Unchecked)
                parent.addChild(child)
            parent.setExpanded(True)
        self.group_builder.set_columns(values)
        old = self.comparison_within.currentData()
        self.comparison_within.clear()
        self.comparison_within.addItem(tr("No stratification"), None)
        for key in sorted(values):
            self.comparison_within.addItem(key, key)
        self.comparison_within.setCurrentIndex(max(0, self.comparison_within.findData(old)))

    def _comparison_items(self):
        """Return every value item of the dimension picker.

        :returns: the checkable value items, in display order.
        """
        tree = self.comparison_dimensions
        return [tree.topLevelItem(index).child(child)
                for index in range(tree.topLevelItemCount())
                for child in range(tree.topLevelItem(index).childCount())]

    def _comparison_selection(self):
        """Read the ticked annotation dimensions and their chosen values.

        :returns: ``(dimensions, levels, within)`` for the comparison API.
        """
        levels = {}
        for item in self._comparison_items():
            if item.checkState(0) == Qt.Checked:
                levels.setdefault(item.parent().text(0), []).append(item.text(0))
        within = self.comparison_within.currentData()
        return [key for key in levels if key != within], {key: values for key, values in levels.items()
                                                           if key != within}, within

    def _comparison_groups(self):
        """The comparison the user defined: explicit built groups, or ticked values.

        :returns: ``(dimensions, levels, within, groups)``; ``groups`` is
            ``None`` when the comparison uses the ticked values.
        """
        dimensions, levels, within = self._comparison_selection()
        groups = self.group_builder.definitions()
        if not groups:
            return dimensions, levels, within, None
        dimensions = [key for key in dict.fromkeys(key for group in groups for key in group) if key != within]
        return dimensions, None, within, groups

    def _await_plate_paint(self):
        """Mark the plate as rebuilt and not yet painted."""
        self._plate_painted = False
        self._plate_built_at = time.monotonic()

    def _prepare_display_after_paint(self):
        """Start the display preparation once the rebuilt plate has painted.

        Its worker reads and samples planes in Python and competes with the
        GUI thread for the interpreter lock; started with the first paint of
        a 384-well plate, the build and paint ran 55-75 ms instead of under
        50. The wait is bounded by :data:`_OVERVIEW_FIRST_PAINT_WAIT_S` for
        a plate that never paints. Its thread is also not started while
        PyTorch's libraries are being mapped
        (:func:`spacr.qt.gil_priority._torch_libraries_mapping`), for at most
        :data:`_OVERVIEW_LIBRARY_WAIT_S`.
        """
        from ..gil_priority import _torch_libraries_mapping

        waited = time.monotonic() - self._plate_built_at
        if self._fields and self.isVisible() and (
                (not self._plate_painted and waited < _OVERVIEW_FIRST_PAINT_WAIT_S)
                or (_torch_libraries_mapping() and waited < _OVERVIEW_LIBRARY_WAIT_S)):
            self._normalization_ready = False
            self._prepare_timer.start(10)
            return
        self._prepare_display()

    def _comparison_request(self):
        """Start a new comparison or representative request and return its number.

        A result delivered for an older number is dropped, whatever its job
        runner reports, so a late answer never replaces a newer one.
        """
        self._comparison_requests = getattr(self, "_comparison_requests", 0) + 1
        return self._comparison_requests

    def _compare_groups(self):
        """Test and plot one measurement across every chosen annotation group."""
        request = self._comparison_request()
        dimensions, levels, within, groups = self._comparison_groups()
        if not dimensions:
            self.status.setText(tr("Tick the annotation values to compare."))
            return
        source = self._metadata_source
        measurement = self.comparison_measurement.currentText().strip()
        folder = Path(source) / "results" / "overview_comparisons"
        names = self.group_builder.names() if groups else None
        def run():
            """Read, test, draw and save the comparison outside the GUI thread.

            :returns: ``(png bytes, saved figure, tables, measurement)`` or an error text.
            """
            from datetime import datetime
            from ...sp_stats import (_compare_image_groups, _image_comparison_figure,
                                     _image_comparison_table, _image_measurement_columns,
                                     _write_image_comparison)

            try:
                frame = _overview_comparison_frame(source)
            except _OverviewValidationError as error:
                return str(error)
            columns = _image_measurement_columns(frame)
            column = measurement or (columns[0] if columns else "")
            stratum = [within] if within else None
            try:
                result = _compare_image_groups(frame, [column], dimensions, groups=groups,
                                               levels=levels, within=stratum, group_names=names)
                stem = "comparison_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                folder.mkdir(parents=True, exist_ok=True)
                tables = _write_image_comparison(result, folder, stem)
                figure, saved = _image_comparison_figure(frame, column, dimensions, groups=groups, levels=levels,
                                                         within=stratum, tests=result["tests"],
                                                         path=str(folder / stem), group_names=names)
            except (KeyError, ValueError, IndexError) as error:
                LOG.warning("overview comparison refused: %s", error)
                return tr("These annotation groups cannot be compared with {measurement}.",
                          measurement=column or tr("Measurement"))
            import tempfile
            import matplotlib.pyplot as plt
            from .figure_queue import render_figure_to_png
            with tempfile.TemporaryDirectory(prefix="spacr-comparison-") as scratch:
                preview = Path(scratch) / "comparison.png"
                png = (preview.read_bytes()
                       if render_figure_to_png(figure, str(preview), write_pdf=False) else b"")
            plt.close(figure)
            return png, str(saved), tables, column, columns, _image_comparison_table(result)
        def shown(result):
            """Show the comparison figure in the image level and offer its measurements.

            :param result: the worker's figure and paths, or a refusal text.
            """
            if request != self._comparison_requests:
                return
            if isinstance(result, str):
                self.status.setText(result)
                return
            png, saved, tables, column, columns, compact = result
            self._show_comparison_table(compact)
            if self.comparison_measurement.count() != len(columns):
                self.comparison_measurement.blockSignals(True)
                self.comparison_measurement.clear()
                self.comparison_measurement.addItems(columns)
                self.comparison_measurement.setCurrentText(column)
                self.comparison_measurement.blockSignals(False)
            pixmap = QPixmap()
            pixmap.loadFromData(png, "PNG")
            self._individual = False
            self._comparison_signature = ("groups", saved)
            self._show_image_panel()
            self.view.set_pixmap(pixmap)
            self._update_image_metadata()
            self.status.setText(tr("Comparison of {measurement}: figure {path}; tables {tables}.",
                                   measurement=column, path=saved, tables=tables["tests"]))
            self._show_representatives(stacked=True)
        self._comparison_signature = None
        self.status.setText(tr("Comparing annotated groups…"))
        self._export_jobs.submit(run, shown)

    def _show_comparison_table(self, frame):
        """Show the compared groups' numbers beside the stacked representative images.

        :param frame: ``spacr.sp_stats._image_comparison_table`` output.
        """
        from .sortable_table import install_sorting, table_item

        columns = [name for name in ("measurement", "group_a", "group_b", "n_a", "n_b", "mean_a", "mean_b",
                                     "log2_ratio", "effect_size", "p_adjusted", "significance")
                   if name in frame.columns]
        table = self.comparison_table
        table.setSortingEnabled(False)
        table.clear()
        table.setColumnCount(len(columns))
        table.setRowCount(len(frame))
        table.setHorizontalHeaderLabels(columns)
        for row, record in enumerate(frame[columns].itertuples(index=False)):
            for column, value in enumerate(record):
                text = f"{value:.4g}" if isinstance(value, float) else str(value)
                table.setItem(row, column, table_item(text))
        install_sorting(table)
        table.resizeColumnsToContents()
        table.setVisible(len(frame) > 0)
        self._show_image_column()

    def _show_representatives(self, stacked=False):
        """Show the most representative image of every chosen annotation class.

        :param stacked: open them as stacked images beside a comparison figure,
            keeping its status, instead of one montage.
        """
        request = self._comparison_requests if stacked else self._comparison_request()
        dimensions, levels, within, groups = self._comparison_groups()
        levels = levels or {}
        by = (["_overview_group"] if groups else dimensions) + ([within] if within else [])
        if not by:
            self.status.setText(tr("Tick the annotation values to compare."))
            return
        method = self.representative_method.currentData()
        measurement = self.comparison_measurement.currentText().strip() or None
        fields, source = self._fields, self._metadata_source
        names = self.group_builder.names() if groups else []
        def run():
            """Rank representatives and pick their source fields off the GUI thread.

            :returns: ``(fields, class captions)`` of the top image per class, or an error text.
            """
            from ...sp_stats import _representative_images

            try:
                frame = _overview_comparison_frame(source)
            except _OverviewValidationError as error:
                return str(error)
            for key, values in levels.items():
                frame = frame[frame[key].astype(str).isin(values)]
            if groups:
                frame = _overview_grouped_frame(frame, groups, names)
            try:
                ranked = _representative_images(frame, by, method=method, measurement=measurement, top=1)
            except (KeyError, ValueError) as error:
                LOG.warning("representative images refused: %s", error)
                return tr("No representative image can be chosen this way for these annotation values.")
            known = {(field.plate, field.well, str(field.field), str(field.time_id), str(field.z_id)): field
                     for field in fields}
            chosen, captions = [], []
            for record in ranked.to_dict("records"):
                field = known.get((str(record["plate"]), str(record["well"]), str(record["field"]),
                                   str(record.get("time_id") or ""), str(record.get("z_id") or "")))
                if field is not None:
                    chosen.append(field)
                    captions.append(" + ".join(str(record[key]) for key in by) + ": " + field.caption)
            return tuple(chosen), tuple(captions)
        def shown(result):
            """Compare the representatives side by side in the image level.

            :param result: the chosen fields and their class captions.
            """
            if request != self._comparison_requests:
                return
            if stacked and (isinstance(result, str) or not result[0]):
                return
            if isinstance(result, str):
                self.status.setText(result)
                return
            chosen, captions = result
            if not chosen:
                self.status.setText(tr("No measured image matches the chosen annotation values."))
                return
            if stacked:
                self._open_image_views(chosen[:_OVERVIEW_MAX_IMAGE_VIEWS])
                return
            self._compare_fields(chosen[:24], tr("Representative images: {classes}",
                                                 classes="; ".join(captions)))
        if not stacked:
            self._comparison_signature = None
            self.status.setText(tr("Choosing representative images…"))
        self._export_jobs.submit(run, shown)

    def _fill_group_choices(self, combo, image=False):
        """Offer physical addresses and every defined well annotation."""
        old = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        if image:
            combo.addItem(tr("Each image independently"), "image")
        for label, key in ((tr("Plate"), "plate"), (tr("Well"), "well"),
                           (tr("Plate row"), "row"), (tr("Plate column"), "column")):
            combo.addItem(label, key)
        keys = {key for values in self._annotation_snapshot().values() for key in values}
        for key in sorted(keys):
            combo.addItem(tr("Metadata: {key}", key=key), key)
        if combo.findData(old) >= 0:
            combo.setCurrentIndex(combo.findData(old))
        combo.blockSignals(False)

    def _editor_options(self):
        """Read one channel editor without changing any displayed transform."""
        method = self.normalization_method.currentData()
        return dict(method=method,
                    low=self.manual_low.value() if method == "manual" else self.advanced_low.value(),
                    high=self.manual_high.value() if method == "manual" else self.advanced_high.value(),
                    baseline=self.background.value(), gamma=self.gamma.value(),
                    reference_channel=self.reference_channel.value(), epsilon=self.reference_epsilon.value())

    def _normalization_options(self):
        """Combine retained global settings with explicitly committed overrides."""
        options = dict(self._global_normalization_options)
        if self._editing_channel is None:
            options = self._editor_options()
        advanced = set(self._editor_options())
        options.update(scope=self.normalization_scope.currentData(),
                       per_channel={str(key): {name: setting for name, setting in value.items() if name in advanced}
                                    for key, value in self._channel_overrides.items()})
        return options

    def _normalization_target_changed(self, *_):
        """Load the effective settings for one channel without changing others.

        Every per-channel value follows the choice: the advanced
        normalization, the percentile stretch, the enhancement steps and the
        ink intensity. Picking a channel shows that channel's values, its own
        where it has them and the global ones otherwise; picking All channels
        shows the global values again. Display settings still waiting for
        Apply are applied first, so they are not taken for the channel's.
        """
        if self._editing_channel is None and callable(self._settings_flush) and not self._loading_controls:
            self._settings_flush()
        if self._editing_channel is None:
            self._global_normalization_options = self._editor_options()
            self._global_display = self._display_editor_values()
        self._editing_channel = self.norm_channel.currentData()
        options = dict(self._global_normalization_options)
        options.update(self._channel_overrides.get(self._editing_channel, {}))
        self._loading_controls = True
        try:
            self.normalization_method.setCurrentIndex(self.normalization_method.findData(options["method"]))
            low, high = ((self.manual_low, self.manual_high) if options["method"] == "manual" else (self.advanced_low, self.advanced_high))
            low.setValue(options["low"])
            high.setValue(options["high"])
            self.background.setValue(options["baseline"])
            self.gamma.setValue(options["gamma"])
            self.reference_channel.setValue(options["reference_channel"])
            self.reference_epsilon.setValue(options["epsilon"])
        finally:
            self._loading_controls = False
        self._load_display_editor()
        self._normalization_control_states()

    def _normalization_control_states(self):
        """Enable only the controls used by the current editor method."""
        method = self.normalization_method.currentData()
        for widget in (self.advanced_low, self.advanced_high):
            widget.setEnabled(method in ("percentile", "log", "asinh", "ratio", "difference"))
        for widget in (self.manual_low, self.manual_high):
            widget.setEnabled(method == "manual")
        self.reference_channel.setEnabled(method in ("ratio", "difference"))
        self.reference_epsilon.setEnabled(method == "ratio")

    def _apply_channel_normalization(self):
        """Commit editor settings to exactly the selected channel.

        The channel keeps the advanced normalization editor's values, as
        before, and every display value -- percentiles, enhancement steps, ink
        intensity -- that differs from the global one; the others keep
        following the global values.
        """
        channel = self.norm_channel.currentData()
        if channel is None:
            self._global_normalization_options = self._editor_options()
        else:
            entry = self._editor_options()
            edited = dict(self._display_editor_values(), **self._display_draft)
            entry.update({name: value for name, value in edited.items()
                          if (name in self._global_display or name.startswith("enhance_"))
                          and value != self._global_display.get(name)})
            self._channel_overrides[int(channel)] = entry
        self._channel_display_changed()
        self._advanced_invalidate()

    def _clear_channel_normalization(self):
        """Reset channel overrides and reload the retained global transform."""
        self._channel_overrides.clear()
        self._normalization_target_changed()
        self._channel_display_changed()
        self._advanced_invalidate()

    def _reset_channel_override(self):
        """Return the selected channel to the global values and show them."""
        channel = self.norm_channel.currentData()
        if channel is None or self._channel_overrides.pop(int(channel), None) is None:
            return
        self._normalization_target_changed()
        self._channel_display_changed()
        self._advanced_invalidate()

    def _display_editor_values(self):
        """The per-channel display values the editor rows show now.

        :returns: ``percentile_low``, ``percentile_high``, ``ink`` and every
            ``enhance_*`` value, as the rows hold them.
        """
        values = {"percentile_low": float(self.low.value()), "percentile_high": float(self.high.value()),
                  "ink": float(self._ink_intensity)}
        values.update(self._enhance_values)
        return values

    def _load_display_editor(self):
        """Show the edited channel's display values in the rows, or the global ones for All channels."""
        values = dict(self._global_display)
        channel = self._editing_channel
        if channel is not None:
            override = self._channel_overrides.get(int(channel), {})
            values.update({name: value for name, value in override.items() if name in values})
        self._display_draft = {}
        if not values:
            return
        self._loading_controls = True
        try:
            self.low.setValue(values["percentile_low"])
            self.high.setValue(values["percentile_high"])
        finally:
            self._loading_controls = False
        if callable(self._write_display_rows):
            self._write_display_rows({name: value for name, value in values.items()
                                      if name == "ink" or name.startswith("enhance_")})
        self._refresh_channel_markers()

    def _channel_display_draft(self, values):
        """Hold row edits made while a channel is picked until Apply settings to channel.

        :param values: ``ink`` and the ``enhance_*`` values the rows hold.
        """
        self._display_draft = dict(values)

    def _set_channel_display_globals(self, ink, enhance_values):
        """Take the global ink intensity and enhancement values from the settings rows.

        :param ink: the global ink intensity for the CMY and CMYK modes.
        :param enhance_values: every ``enhance_*`` setting's global value.
        """
        ink = float(ink if ink is not None else 1.)
        enhance_values = dict(enhance_values)
        if ink == self._ink_intensity and enhance_values == self._enhance_values:
            return
        self._ink_intensity, self._enhance_values = ink, enhance_values
        if self._editing_channel is None:
            self._global_display = self._display_editor_values()
        self._channel_display_changed()

    def _channel_display_options(self, view=False):
        """The per-channel display values as :func:`_overview_pixels` takes them.

        :param view: also carry the per-channel enhancement chains, for the
            on-screen views; exports never receive enhancement.
        :returns: ``channel_percentiles``, ``ink_intensity`` and, for views,
            ``channel_enhancement``; each only when some channel differs from
            the global value, so the plain views keep their cache keys.
        """
        options = {}
        percentiles, inks, chains = {}, {}, {}
        for channel, entry in self._channel_overrides.items():
            if "percentile_low" in entry or "percentile_high" in entry:
                low = float(entry.get("percentile_low", self._global_display.get("percentile_low", self.low.value())))
                high = float(entry.get("percentile_high", self._global_display.get("percentile_high", self.high.value())))
                if 0 <= low < high <= 100:
                    percentiles[channel] = (low, high)
            if "ink" in entry:
                inks[channel] = float(entry["ink"])
            enhance = {name: value for name, value in entry.items() if name.startswith("enhance_")}
            if enhance and view:
                try:
                    chains[channel] = _overview_chain(dict(self._enhance_values, **enhance))
                except ValueError as error:
                    self.status.setText(tr("Channel {channel} enhancement is not applied: {error}",
                                           channel=channel, error=error))
        if percentiles:
            options["channel_percentiles"] = percentiles
        if inks or self._ink_intensity != 1.:
            try:
                shown = self._choices()[0]
            except (ValueError, TypeError):
                shown = tuple(inks)
            options["ink_intensity"] = {channel: inks.get(channel, self._ink_intensity)
                                        for channel in dict.fromkeys(shown + tuple(inks))}
        if chains:
            options["channel_enhancement"] = chains
        return options

    def _refresh_channel_markers(self):
        """Mark the channels with their own settings in the Channel to edit choice."""
        for index in range(self.norm_channel.count()):
            channel = self.norm_channel.itemData(index)
            if channel is None:
                continue
            text = (tr("{channel} (own settings)", channel=channel)
                    if int(channel) in self._channel_overrides else str(channel))
            if self.norm_channel.itemText(index) != text:
                self.norm_channel.setItemText(index, text)
        if hasattr(self, "_reset_channel_button"):
            channel = self.norm_channel.currentData()
            self._reset_channel_button.setEnabled(channel is not None and int(channel) in self._channel_overrides)

    def _channel_display_changed(self):
        """Redraw every view after a per-channel display value changed."""
        self._refresh_channel_markers()
        self._refresh_scope_limits()
        self._display_generation += 1
        self._cache.clear()
        self._thumbnail_cache.clear()
        self._displayed_montage = None
        self._displayed_field = None
        self._request_visuals()
        self._reload_image_views()
        item = self.fields.currentItem()
        if item is not None and self._individual:
            self._field(item, preserve_view=True)

    def _build_advanced_controls(self, column):
        """Keep independent channel editing distinct from the existing reference modes."""
        from PySide6.QtWidgets import QSpinBox, QDoubleSpinBox, QGridLayout
        from .setup_dialog import _SetupComboBox
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        self.advanced = Toggle(tr("Advanced channel normalization"))
        layout.addWidget(self.advanced)
        self.normalization_method, self.normalization_scope = _SetupComboBox(), _SetupComboBox()
        for text, key in ((tr("Percentiles"), "percentile"), (tr("Minimum / maximum"), "minmax"),
            (tr("Fixed intensity bounds"), "manual"), (tr("Mean ± 3 standard deviations"), "zscore"),
            (tr("Log intensity"), "log"), (tr("Inverse hyperbolic sine"), "asinh"),
            (tr("Raw display values"), "none"), (tr("Ratio to reference channel"), "ratio"),
            (tr("Difference from reference channel"), "difference")):
            self.normalization_method.addItem(text, key)
        self._fill_group_choices(self.normalization_scope, image=True)
        self.advanced_low, self.advanced_high = QDoubleSpinBox(), QDoubleSpinBox()
        for control, value in ((self.advanced_low, 1), (self.advanced_high, 99)):
            control.setRange(0, 100)
            control.setValue(value)
        self.manual_low, self.manual_high = QDoubleSpinBox(), QDoubleSpinBox()
        self.background, self.gamma = QDoubleSpinBox(), QDoubleSpinBox()
        for control in (self.manual_low, self.manual_high, self.background):
            control.setRange(-1e12, 1e12)
            control.setDecimals(5)
        self.manual_high.setValue(255)
        self.gamma.setRange(.01, 20)
        self.gamma.setValue(1)
        self.reference_channel, self.reference_epsilon = QSpinBox(), QDoubleSpinBox()
        self.reference_channel.setRange(0, 65535)
        self.reference_epsilon.setDecimals(12)
        self.reference_epsilon.setRange(1e-12, 1e12)
        self.reference_epsilon.setValue(1e-8)
        self.reference_epsilon.setToolTip(tr("Ratio keeps pixels only when abs(reference) > epsilon; excluded pixels display black."))
        from .flow import FlowHost, FlowLayout
        host = FlowHost(widget)
        grid = FlowLayout(host)
        choices = ((tr("Method"), self.normalization_method), (tr("Reference group"), self.normalization_scope),
            (tr("Lower percentile"), self.advanced_low), (tr("Upper percentile"), self.advanced_high),
            (tr("Fixed low"), self.manual_low), (tr("Fixed high"), self.manual_high),
            (tr("Subtract background"), self.background), (tr("Gamma"), self.gamma),
            (tr("Reference channel (ratio / difference)"), self.reference_channel),
            (tr("Minimum absolute reference value"), self.reference_epsilon))
        for index, (label, control) in enumerate(choices):
            pair = QWidget(host)
            pair_layout = QVBoxLayout(pair)
            pair_layout.setContentsMargins(0, 0, 0, 0)
            caption = QLabel(label)
            caption.setWordWrap(True)
            pair_layout.addWidget(caption)
            pair_layout.addWidget(control)
            grid.addWidget(pair)
        layout.addWidget(host)
        self.norm_channel = _SetupComboBox()
        self.norm_channel.addItem(tr("All channels"), None)
        self._editing_channel = None
        self._global_normalization_options = self._editor_options()
        apply = QPushButton(tr("Apply settings to channel"))
        apply.clicked.connect(self._apply_channel_normalization)
        reset = QPushButton(tr("Reset channel overrides"))
        reset.clicked.connect(self._clear_channel_normalization)
        self._reset_channel_button = QPushButton(tr("Reset this channel"))
        self._reset_channel_button.setToolTip(tr("Return the selected channel to the global values."))
        self._reset_channel_button.setEnabled(False)
        self._reset_channel_button.clicked.connect(self._reset_channel_override)
        self._channel_override_buttons = (apply, self._reset_channel_button, reset)
        override_host = FlowHost(widget)
        row = FlowLayout(override_host)
        for control in (self.norm_channel, apply, self._reset_channel_button, reset):
            row.addWidget(control)
        layout.addWidget(override_host)
        column.addWidget(widget)
        channel_host = FlowHost(self)
        self._channel_host = channel_host
        channel_row = FlowLayout(channel_host)
        self._channel_buttons = {}
        self._channel_bar = channel_row
        channel_row.addWidget(QLabel(tr("Channels")))
        column.addWidget(channel_host)
        measurements = QHBoxLayout()
        self.workers = QSpinBox()
        self.workers.setRange(1, max(1, os.cpu_count() or 1))
        self.workers.setValue(min(4, self.workers.maximum()))
        measure = QPushButton(tr("Export image-level measurements"))
        measure.clicked.connect(self._measure)
        for control in (QLabel(tr("Parallel images")), self.workers, measure):
            measurements.addWidget(control)
        column.addLayout(measurements)
        column.addWidget(QLabel(tr("PDFs: results/ · Measurements: measurements/measurements.db → fov")))
        self.advanced.toggled.connect(self._advanced_invalidate)
        self.norm_channel.currentIndexChanged.connect(self._normalization_target_changed)
        for control in (self.normalization_method, self.normalization_scope):
            control.currentIndexChanged.connect(self._advanced_changed)
        for control in (self.advanced_low, self.advanced_high, self.manual_low, self.manual_high,
                        self.background, self.gamma, self.reference_channel, self.reference_epsilon):
            control.valueChanged.connect(self._advanced_changed)

    def _advanced_changed(self, *_):
        """Keep channel drafts isolated until explicitly applied."""
        if self._loading_controls:
            return
        self._normalization_control_states()
        if self._editing_channel is None:
            self._global_normalization_options = self._editor_options()
        elif self.sender() is not self.normalization_scope:
            return
        self._advanced_invalidate()

    def _advanced_invalidate(self, *_):
        """Recompute advanced limits without changing the saved reference mode."""
        self._advanced_model = None
        self._cache.clear()
        self._thumbnail_cache.clear()
        self._displayed_montage = None
        self._prepare_display()

    def _annotation_snapshot(self):
        """Capture arbitrary metadata with current legacy condition assignments."""
        values = {address: dict(metadata) for address, metadata in self._annotations.items()}
        for address, condition in self._conditions.items():
            values.setdefault(address, {})["condition"] = condition
        for address in values:
            if address not in self._conditions:
                values[address].pop("condition", None)
        return values

    def _annotation_name(self):
        """Use stable keys for standard metadata and literal custom names."""
        if self.annotation_key.currentIndex() >= 0 and self.annotation_key.currentText() == self.annotation_key.itemText(self.annotation_key.currentIndex()):
            return self.annotation_key.currentData()
        return self.annotation_key.currentText().strip()

    def _persist_metadata(self):
        """Serialize pending per-source metadata snapshots, including final close edits."""
        import json
        if not self._metadata_source:
            return
        self._metadata_revision += 1
        destination = Path(self._metadata_source) / "results" / "overview_annotations.json"
        payload = dict(wells=[dict(plate=plate, well=well, metadata=metadata,
                                   plate_key=list(self._plate_key(plate)))
            for (plate, well), metadata in sorted(self._annotation_snapshot().items())])
        with self._annotation_write_lock:
            self._annotation_pending_snapshot[destination] = payload
            if self._annotation_write_running:
                return
            self._annotation_write_running = True
        def write():
            """Drain immutable snapshots outside Qt and publish atomically."""
            import json
            from ... import tabular
            try:
                while True:
                    with self._annotation_write_lock:
                        if not self._annotation_pending_snapshot:
                            self._annotation_write_running = False
                            return
                        target = next(iter(self._annotation_pending_snapshot))
                        values = self._annotation_pending_snapshot.pop(target)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    tabular._publish(str(target), lambda pending:
                        Path(pending).write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8"))
            except BaseException:
                with self._annotation_write_lock:
                    self._annotation_write_running = False
                raise
        self._metadata_jobs.submit(write)

    def _restore_metadata(self):
        """Load saved metadata only for the indexed source's physical wells."""
        if not self._metadata_source:
            return
        source, fields, cancelled = self._metadata_source, self._fields, self._cancelled
        revision = self._metadata_revision
        known = {(field.plate, field.well) for field in fields if not field.unresolved}
        by_key = {_overview_plate_key(field): field.plate for field in fields}
        def read():
            """Validate source-relative metadata without GUI-thread file access.

            :mod:`spacr.measure` is imported only when a Measure database can
            exist: a fresh source has none, and the import would otherwise wait
            behind the pipeline preload's library imports before any metadata
            reached the overview.
            """
            import json

            values = {}
            candidate = Path(source).expanduser()
            if candidate.suffix == ".db" or (candidate / "measurements" / "measurements.db").is_file():
                from ...measure import _image_annotation_database, _well_image_annotations
                database = _image_annotation_database(source)
            else:
                database = None
            if database is not None and database.is_file():
                for address, metadata in _well_image_annotations(database).items():
                    if address in known:
                        values[address] = {str(key): str(value) for key, value in metadata.items()
                                           if key != "_mixed"}
            path = Path(source) / "results" / "overview_annotations.json"
            if not path.is_file():
                return values
            records = json.loads(path.read_text(encoding="utf-8")).get("wells", ())
            stored, values = values, {}
            for record in records:
                plate = by_key.get(tuple(str(part) for part in record.get("plate_key") or ()), record["plate"])
                address = str(plate), str(record["well"])
                if address not in known or address in values or not isinstance(record["metadata"], dict):
                    raise _OverviewValidationError(tr("Annotation table identities are missing, duplicated or outside this source."))
                values[address] = {str(key): str(value) for key, value in record["metadata"].items()}
            for address, metadata in stored.items():
                values.setdefault(address, {}).update(metadata)
            return values
        def loaded(values):
            """Install only metadata belonging to the unchanged indexed generation."""
            if (cancelled.is_set() or fields is not self._fields or revision != self._metadata_revision
                    or not values):
                return
            for address, metadata in values.items():
                self._annotations.setdefault(address, {}).update(metadata)
                if metadata.get("condition") and address not in self._conditions:
                    self._conditions[address] = metadata["condition"]
            self._fill_group_choices(self.grouping)
            self._fill_group_choices(self.normalization_scope, image=True)
            self._refresh_comparison_choices()
            self._filter()
            if self.advanced.isChecked():
                self._prepare_display()
        self._metadata_jobs.submit(read, loaded)

    def _remember_wells(self):
        """Retain plate-qualified selections, across every shown plate, and each plate's acquisition context."""
        for address in [address for address in self._well_positions if address[0] in self._plate_tables]:
            self._selected_wells.discard(address)
            self._selected_well_axes.pop(address, None)
            self._selected_well_options.pop(address, None)
        for plate, table in self._plate_tables.items():
            for item in table.selectedItems():
                address = plate, item.data(Qt.UserRole)
                self._selected_wells.add(address)
                self._selected_well_axes[address] = (self.time_axis.currentData(), self.z_axis.currentData())
                self._selected_well_options[address] = self._axis_options()

    def _clicked_well(self, item):
        """Open ordinary clicks while retaining drag and modifier selections.

        A click on a collapsed row's or column's strip shows it again instead.
        """
        rows, columns = self._collapsed_lines(item.tableWidget()._overview_plate)
        if item.row() in rows or item.column() in columns:
            rows.discard(item.row())
            columns.discard(item.column())
            self._relayout_collapsed()
            self._persist_view_state()
            return
        if not self._drag_selected and not getattr(self, "_press_modifiers", Qt.NoModifier) & (Qt.ControlModifier | Qt.ShiftModifier):
            self._well(item)

    def _channel_toggled(self, *_):
        """Toggle source channel IDs independently without changing their meaning."""
        values = [str(channel) for channel, widget in self._channel_buttons.items() if widget.isChecked()]
        self.channels.setText(",".join(values))
        self._refresh_image()

    def _install_channel_buttons(self, info):
        """Expose actual declared channels without decoding image pixels."""
        selected = tuple(int(value.strip()) for value in self.channels.text().split(",") if value.strip().isdigit())
        available = tuple(info["channels"])
        if tuple(self._channel_buttons) == available:
            for channel, control in self._channel_buttons.items():
                control.blockSignals(True)
                control.setChecked(channel in selected)
                control.blockSignals(False)
            return
        for widget in self._channel_buttons.values():
            self._channel_bar.removeWidget(widget)
            widget.deleteLater()
        self._channel_buttons.clear()
        self.norm_channel.blockSignals(True)
        self.norm_channel.clear()
        self.norm_channel.addItem(tr("All channels"), None)
        names = self._overview_options.get("channel_names") or {}
        for channel in available:
            name = names.get(channel, names.get(str(channel), ""))
            control = Toggle(f"{channel} {name}".strip())
            control.setChecked(channel in selected)
            control.setToolTip(tr("Show channel {channel}", channel=channel))
            control.toggled.connect(self._channel_toggled)
            self._channel_bar.addWidget(control)
            self._channel_buttons[channel] = control
            self.norm_channel.addItem(str(channel), channel)
        self.norm_channel.blockSignals(False)
        self._normalization_target_changed()
        self._refresh_channel_markers()
        self._channels_found.emit(available)

    def _sample_fields(self, fields):
        """Sample physical groups or arbitrary metadata reproducibly."""
        grouping = self.grouping.currentData()
        if grouping in ("plate", "row", "column", "well"):
            return _overview_sample(fields, self.count.value(), self.seed.value(), grouping, self.random.isChecked())
        import random
        groups, result = {}, []
        annotations = self._annotation_snapshot()
        for field in sorted(fields, key=lambda value: repr(value.identity)):
            key = _overview_normalization_group(field, grouping, annotations)
            groups.setdefault(key, []).append(field)
        rng = random.Random(self.seed.value())
        for key in sorted(groups):
            values = groups[key]
            count = min(self.count.value(), len(values))
            result.extend(rng.sample(values, count) if self.random.isChecked() else values[:count])
        return tuple(result)

    def _measure(self):
        """Run selected raw image measurements separately from display and PDF jobs."""
        import threading
        if not self._metadata_source or not self._geometry_valid:
            self.status.setText(tr("Choose an image folder."))
            return
        try:
            channels, limits = self._choices()
        except (ValueError, TypeError) as error:
            self.status.setText(str(error))
            return
        if self._measurement_jobs.is_busy():
            self.status.setText(tr("Wait for the current measurement run or cancel it."))
            return
        fields = self._visible if self.export_scope.currentData() == "view" and not self.use_selection.isChecked() else self._export_selection()
        if not fields:
            self.status.setText(tr("Select fields or load a plate before exporting."))
            return
        self._measurement_cancelled = threading.Event()
        measurements = self._measurement_settings() if callable(self._measurement_settings) else None
        source = self._metadata_source
        workers = None if measurements is not None else self.workers.value()
        annotations, cancelled = self._annotation_snapshot(), self._measurement_cancelled
        choices = self._display_kwargs()
        axes = {key: choices[key] for key in ("time_index", "z_mode", "z_index")}
        axes["field_options"] = self._selection_options(fields)
        def done(report):
            """Report atomic database publication and per-image diagnostics."""
            if report["cancelled"]:
                self.status.setText(tr("Measurements cancelled; existing database rows were preserved."))
            elif report["errors"]:
                self.status.setText(tr("Measurements were not published: {count} images failed. {error}",
                    count=len(report["errors"]), error=report["errors"][0]["error"]))
            else:
                message = tr("Saved {count} image measurements to {path} (table fov).",
                             count=report["count"], path=report["path"])
                if report.get("exports"):
                    message += " " + tr("Exported: {files}", files=", ".join(report["exports"].values()))
                self.status.setText(message)
        if workers is None:
            self.status.setText(tr("Starting workers… measuring {count} image planes; "
                                   "parallel workers start a few seconds apart.", count=len(fields)))
        else:
            self.status.setText(tr("Measuring {count} image planes with {workers} parallel workers…",
                                   count=len(fields), workers=workers))
        progress = self.measurement_progress.emit
        self._measurement_jobs.submit(lambda: _overview_measurements_export(fields, source, channels,
            workers, cancelled, annotations, progress, measurements=measurements, **axes), done)

    def _show_measurement_progress(self, completed, total):
        """Report how many images the running measurement has finished.

        :param completed: images measured so far.
        :param total: images in the run.
        """
        if self._measurement_jobs.is_busy():
            self.status.setText(tr("Measured {completed} of {total} images…",
                                   completed=completed, total=total))

    def _import_annotation_scheme(self, *_):
        """Ask for a well annotation scheme, preview it, and apply it to this source's wells.

        Read by :func:`spacr.measure._read_well_annotation_scheme`: a list with
        a well column (and optionally plate), one annotation per other column,
        or a plate map named after its file. Written by
        :func:`spacr.measure._import_well_annotation_scheme` as undoable
        annotation operations on every image of the matched wells.
        """
        from PySide6.QtWidgets import QFileDialog

        if self._export_jobs.is_busy():
            self.status.setText(tr("Wait for the current export to finish or cancel it."))
            return
        path, _ = QFileDialog.getOpenFileName(self, tr("Import annotation scheme"), "",
                                              tr("Tables (*.csv *.tsv *.txt *.xlsx)"))
        if not path or not self._metadata_source:
            return
        self._export_jobs.submit(functools.partial(_overview_scheme_preview, path),
                                 functools.partial(self._scheme_previewed, self._fields, path))

    def _scheme_previewed(self, fields, path, frame):
        """Show what a scheme would annotate and apply it once confirmed.

        :param fields: the indexed fields the scheme was read for.
        :param path: the scheme file.
        :param frame: the scheme, one row per well.
        """
        from PySide6.QtWidgets import QMessageBox

        if fields is not self._fields:
            return
        annotations, unmatched = _overview_scheme_annotations(frame, fields)
        columns = ", ".join(str(column) for column in list(frame.columns)[2:])
        answer = QMessageBox.question(self, tr("Import annotation scheme"), tr(
            "Annotate {wells} wells of this source with {columns} from {name}? "
            "{unmatched} wells of the scheme are not in this source.",
            wells=len(annotations), columns=columns, name=Path(path).name,
            unmatched=unmatched))
        if answer != QMessageBox.Yes:
            return
        records = [field.__dict__ for field in fields if not field.unresolved]
        self._export_jobs.submit(
            functools.partial(_overview_import_scheme, self._metadata_source, path, records),
            functools.partial(self._scheme_imported, fields, annotations))

    def _scheme_imported(self, fields, annotations, report):
        """Show an imported scheme's annotations in this browser.

        :param fields: the indexed fields it was imported for.
        :param annotations: ``{(plate, well): {column: value}}`` from the preview.
        :param report: :func:`spacr.measure._import_well_annotation_scheme`'s summary.
        """
        if fields is not self._fields:
            return
        for address, values in annotations.items():
            self._annotations.setdefault(address, {}).update(values)
            if "condition" in values:
                self._conditions[address] = values["condition"]
        self._persist_metadata()
        self._fill_group_choices(self.grouping)
        self._fill_group_choices(self.normalization_scope, image=True)
        self._refresh_comparison_choices()
        self._filter()
        message = tr("Imported {columns} for {images} images in {wells} wells.",
                     columns=", ".join(report["columns"]), images=report["images"], wells=report["wells"])
        if report["unmatched"]:
            message += " " + tr("{count} wells of the scheme are not in this source.",
                                count=len(report["unmatched"]))
        self.status.setText(message)

    def _save_annotations(self):
        """Write conditions only to the user's explicitly chosen output table."""
        from PySide6.QtWidgets import QFileDialog
        self._annotation_file_actions += 1
        if self._export_jobs.is_busy():
            self.status.setText(tr("Wait for the current export to finish or cancel it."))
            return
        path, _ = QFileDialog.getSaveFileName(self, tr("Save condition annotations"),
                                             "", tr("CSV tables (*.csv)"))
        if not path:
            return
        records = tuple((plate, well, value) for (plate, well), value in sorted(self._conditions.items()))
        def write():
            """Atomically write the captured table without changing source images.

            :returns: the requested output path.
            """
            import pandas as pd
            from ... import tabular
            tabular.write_table(pd.DataFrame(records, columns=("plate", "well", "condition")),
                                path, canonicalise=False)
            return path
        self._export_jobs.submit(write, lambda output: self.status.setText(
            tr("Condition annotations saved: {path}", path=output)))

    def _load_annotations(self):
        """Validate every loaded plate/well identity before replacing conditions."""
        from PySide6.QtWidgets import QFileDialog
        self._annotation_file_actions += 1
        if self._export_jobs.is_busy():
            self.status.setText(tr("Wait for the current export to finish or cancel it."))
            return
        path, _ = QFileDialog.getOpenFileName(self, tr("Load condition annotations"),
                                             "", tr("CSV tables (*.csv)"))
        if not path:
            return
        fields, cancelled = self._fields, self._cancelled
        revision = self._metadata_revision
        known = {(field.plate, field.well) for field in fields if not field.unresolved}
        def read():
            """Read a real table and refuse foreign or duplicate physical identities.

            :returns: a validated immutable sequence of condition assignments.
            """
            from ... import tabular
            table = tabular.read_table(path, canonicalise=False, dtype=str, keep_default_na=False)
            if set(table.columns) != {"plate", "well", "condition"}:
                raise _OverviewValidationError(tr("Annotation tables must contain plate, well and condition columns."))
            result = {}
            for record in table.fillna("").to_dict("records"):
                key = str(record["plate"]), str(record["well"])
                if key not in known or key in result:
                    raise _OverviewValidationError(tr("Annotation table identities are missing, duplicated or outside this source."))
                result[key] = str(record["condition"])
            return tuple(result.items())
        def loaded(records):
            """Install validated assignments only for their original source generation.

            :param records: validated physical identities and condition strings.
            """
            if not cancelled.is_set() and fields is self._fields and revision == self._metadata_revision:
                self._conditions = dict(records)
                self._persist_metadata()
                self._fill_group_choices(self.grouping)
                self._fill_group_choices(self.normalization_scope, image=True)
                self._refresh_comparison_choices()
                self._filter()
                if self._overview_options.get("normalization") or self.advanced.isChecked():
                    self._prepare_display()
        self._export_jobs.submit(read, loaded)

    def _update_field_thumbnails(self, signature, channels, limits, options):
        """Decode one visible field tile at a time, one per open well first.

        The next tile comes from the open well view with the fewest current
        tiles, so every stacked well shows an image before any well gets its
        second one.

        :param signature: the current source, axes and display generation.
        :param channels: exact requested source channel identities.
        :param limits: optional percentile limits.
        :param options: the immutable normalization and acquisition selection.
        """
        if self._field_thumbnail_jobs.is_busy():
            return
        cancelled, choices = self._cancelled, []
        for order, fields in enumerate(self._field_lists()):
            if not fields.isVisible():
                continue
            viewport = fields.viewport()
            ratio = device_ratio(viewport)
            edge = (None if self._quality_original()
                    else min(math.ceil(self._tile_size * ratio), self._quality_cap()))
            loaded, waiting = 0, []
            for index in range(fields.count()):
                item = fields.item(index)
                if not viewport.rect().intersects(fields.visualItemRect(item)):
                    continue
                key = (item.data(Qt.UserRole).identity, channels, limits,
                       self._display_options_key(options), ratio, edge)
                if item.data(Qt.UserRole + 1) == key:
                    loaded += 1
                else:
                    waiting.append((item.data(Qt.UserRole), key))
            if waiting:
                choices.append((loaded, order, fields, edge, *waiting[0]))
        if not choices:
            return
        _loaded, _order, owner, edge, field, key = min(choices, key=lambda choice: choice[:2])
        def read():
            """Decode only the requested tile, preserving unreadable-field state.

            :returns: detached pixels and an empty error, or None and the refusal.
            """
            try:
                return _overview_pixels(field, channels, limits,
                                         edge=edge, **options), ""
            except Exception as exc:
                return None, str(exc)
        def shown(result):
            """Install only an exact current visible field identity.

            :param result: detached worker pixels and an optional refusal message.
            """
            from PySide6.QtGui import QIcon
            if cancelled.is_set() or self._visual_context() != signature or owner not in self._field_lists():
                return
            pixels, error = result
            if error:
                self.status.setText(tr("Could not read {field}: {error}", field=field.caption, error=error))
            for current_index in range(owner.count()):
                current = owner.item(current_index)
                if current.data(Qt.UserRole).identity != field.identity:
                    continue
                current.setData(Qt.UserRole + 1, key)
                current.setData(Qt.UserRole + 3, None if pixels is None else tuple(pixels.shape[:2]))
                if pixels is None:
                    current.setIcon(QIcon())
                else:
                    image = QImage(pixels.data, pixels.shape[1], pixels.shape[0],
                                   pixels.strides[0], QImage.Format_RGB888).copy()
                    pixmap = scaled_for(QPixmap.fromImage(image), owner.viewport(), owner.iconSize())
                    current.setIcon(QIcon(pixmap))
                break
        self._field_thumbnail_jobs.submit(read, shown)

    def _zoom_plate(self, delta=0):
        """Fit every shown plate in its own view, or explicitly magnify the cells of every plate.

        :param delta: signed magnification step, or zero to restore whole-plate fit.
        """
        self._plate_fit = not bool(delta)
        tables = self._plate_table_list()
        if delta and self._plate_cell_extent is not None:
            factor = 1.25 if delta > 0 else .8
            extent = tuple(max(2, min(512, round(value * factor))) for value in self._plate_cell_extent)
            self._plate_extents = {table: extent for table in tables}
        else:
            self._plate_extents = {table: self._fitted_extent(table) for table in tables}
        policy = Qt.ScrollBarAlwaysOff if self._plate_fit else Qt.ScrollBarAsNeeded
        for table in tables:
            table.setHorizontalScrollBarPolicy(policy)
            table.setVerticalScrollBarPolicy(policy)
        self._plate_cell_extent = self._plate_extents.get(self.wells, self._plate_extents[tables[0]])
        self._apply_plate_extent()

    def _fitted_extent(self, table):
        """The well size that fits ``table``'s whole plate, collapsed lines as strips, in its view."""
        columns, rows = max(1, table.columnCount()), max(1, table.rowCount())
        folded_rows, folded_columns = (
            sum(1 for index in lines if index < count)
            for lines, count in zip(self._collapsed_lines(table._overview_plate), (rows, columns)))
        strip = _OVERVIEW_COLLAPSED_STRIP
        width = max(1, (table.viewport().width() - strip * folded_columns)
                    // max(1, columns - folded_columns))
        height = max(1, (table.viewport().height() - strip * folded_rows)
                     // max(1, rows - folded_rows))
        return width, height

    def _apply_plate_extent(self):
        """Restore the same native well size on every shown plate after source-axis or annotation changes."""
        from PySide6.QtCore import QSize

        strip = _OVERVIEW_COLLAPSED_STRIP
        for table in self._plate_table_list():
            width, height = self._plate_extents.get(table, self._plate_cell_extent)
            rows, columns = self._collapsed_lines(table._overview_plate)
            table._overview_collapsed = (frozenset(rows), frozenset(columns))
            table.setIconSize(QSize(max(1, width - 2), max(1, height - 2)))
            for column in range(table.columnCount()):
                table.setColumnWidth(column, strip if column in columns else width)
            for row in range(table.rowCount()):
                table.setRowHeight(row, strip if row in rows else height)
            self._label_collapsed_lines(rows, columns, table)
            table.viewport().update()
        self._request_visuals()

    def _collapsed_lines(self, plate=None):
        """A plate's collapsed rows and columns.

        :param plate: the plate's session name; the shown plate when ``None``.
        :returns: ``(rows, columns)``, two sets of zero-based table indices
            kept per plate, so each plate remembers its own.
        """
        plate = self.plates.currentData() if plate is None else plate
        if plate is None:
            return set(), set()
        return self._collapsed.setdefault(plate, (set(), set()))

    def _line_name(self, axis, index):
        """The header name of plate row or column ``index``, such as ``B`` or ``3``."""
        item = (self.wells.verticalHeaderItem(index) if axis == "row"
                else self.wells.horizontalHeaderItem(index))
        if item is None:
            return str(index + 1)
        name = item.data(Qt.UserRole)
        return str(item.text() if name is None else name)

    def _label_collapsed_lines(self, rows, columns, table=None):
        """Blank collapsed headers and give every header its collapse help.

        :param table: the plate table; the shown plate's when ``None``.
        """
        table = self.wells if table is None else table
        for axis, count, item_of, lines in (
                ("row", table.rowCount(), table.verticalHeaderItem, rows),
                ("column", table.columnCount(), table.horizontalHeaderItem, columns)):
            for index in range(count):
                item = item_of(index)
                if item is None:
                    continue
                if item.data(Qt.UserRole) is None:
                    item.setData(Qt.UserRole, item.text())
                name = str(item.data(Qt.UserRole))
                folded = index in lines
                item.setText("" if folded else name)
                if axis == "row":
                    item.setToolTip(tr("Row {row} is collapsed; double-click it or its strip to show it again.", row=name)
                                    if folded else
                                    tr("Double-click to collapse row {row} and bring the other rows closer; "
                                       "right-click for more.", row=name))
                else:
                    item.setToolTip(tr("Column {column} is collapsed; double-click it or its strip to show it again.",
                                       column=name) if folded else
                                    tr("Double-click to collapse column {column} and bring the other columns closer; "
                                       "right-click for more.", column=name))

    def _relayout_collapsed(self):
        """Close the plate up around the collapsed rows and columns, or open it again."""
        if self._plate_fit or self._plate_cell_extent is None:
            self._zoom_plate()
        else:
            self._apply_plate_extent()

    def _set_line_collapsed(self, axis, index, collapsed, *_):
        """Collapse plate rows or columns to thin strips, or show them again.

        Only the table geometry changes: selection, annotations, comparison
        groups and measurements are untouched.

        :param axis: ``"row"`` or ``"column"``.
        :param index: one zero-based table index, or several.
        :param collapsed: True collapses, False shows them again.
        """
        rows, columns = self._collapsed_lines()
        lines = rows if axis == "row" else columns
        for line in ((index,) if isinstance(index, int) else tuple(index)):
            if collapsed:
                lines.add(int(line))
            else:
                lines.discard(int(line))
        self._relayout_collapsed()
        self._persist_view_state()

    def _selected_lines(self, axis):
        """The rows or columns of the shown plate that hold a selected well."""
        return {index.row() if axis == "row" else index.column() for index in self.wells.selectedIndexes()}

    def _line_targets(self, axis, index, selected=None):
        """The lines one row or column action applies to.

        A line that belongs to a selection of several rows or columns takes
        the whole selection with it; otherwise only the line itself.

        :param axis: ``"row"`` or ``"column"``.
        :param index: the clicked line.
        :param selected: the selected lines, read from the table when ``None``.
        :returns: the zero-based indices, sorted.
        """
        selected = self._selected_lines(axis) if selected is None else set(selected)
        return sorted(selected) if index in selected and len(selected) > 1 else [index]

    def _toggle_line(self, axis, index):
        """Collapse an open plate row or column, or show a collapsed one, with the rest of its selection.

        A double click on a header first selects that line alone, so the
        selection read at the press before it decides which lines follow.
        """
        if index < 0:
            return
        snapshot = getattr(self, "_header_press_lines", None)
        self._header_press_lines = None
        selected = snapshot[1] if snapshot is not None and snapshot[0] == axis else None
        lines = self._collapsed_lines()[0 if axis == "row" else 1]
        self._set_line_collapsed(axis, self._line_targets(axis, index, selected), index not in lines)

    def _show_all_lines(self, *_):
        """Show every collapsed row and column of the shown plate again."""
        rows, columns = self._collapsed_lines()
        rows.clear()
        columns.clear()
        self._relayout_collapsed()
        self._persist_view_state()

    def _line_actions(self, menu, axis, index):
        """Add the collapse or show action for one row or column to ``menu``."""
        targets = self._line_targets(axis, index)
        name = ", ".join(self._line_name(axis, line) for line in targets)
        folded = index in self._collapsed_lines()[0 if axis == "row" else 1]
        several = len(targets) > 1
        if axis == "row" and several:
            caption = tr("Show rows {rows}", rows=name) if folded else tr("Collapse rows {rows}", rows=name)
        elif axis == "row":
            caption = tr("Show row {row}", row=name) if folded else tr("Collapse row {row}", row=name)
        elif several:
            caption = (tr("Show columns {columns}", columns=name) if folded
                       else tr("Collapse columns {columns}", columns=name))
        else:
            caption = (tr("Show column {column}", column=name) if folded
                       else tr("Collapse column {column}", column=name))
        menu.addAction(caption).triggered.connect(
            functools.partial(self._set_line_collapsed, axis, tuple(targets), not folded))

    def _show_all_action(self, menu):
        """Add "Show all rows and columns", enabled while anything is collapsed."""
        action = menu.addAction(tr("Show all rows and columns"))
        action.setEnabled(any(self._collapsed_lines()))
        action.triggered.connect(self._show_all_lines)

    def _header_menu(self, axis, position):
        """Offer collapsing a plate row or column from its header.

        :param axis: ``"row"`` or ``"column"``.
        :param position: the click position in header coordinates.
        :returns: the shown menu.
        """
        from PySide6.QtWidgets import QMenu

        header = self.wells.verticalHeader() if axis == "row" else self.wells.horizontalHeader()
        index = header.logicalIndexAt(position)
        menu = QMenu(header)
        if index >= 0:
            self._line_actions(menu, axis, index)
        self._show_all_action(menu)
        self._header_context_menu = menu
        menu.popup(header.mapToGlobal(position))
        return menu

    def _open_selected_well(self):
        """Open the selected well without conflating selection with navigation."""
        item = self.wells.currentItem()
        if item is not None:
            self._well(item)

    def _wrap_controls(self, column, splitter):
        """Keep essential navigation visible and put detailed controls in optional scrolling.

        :param column: the existing dialog layout.
        :param splitter: image workspace outside the configuration scroller.
        """
        from PySide6.QtWidgets import QScrollArea
        from .flow import FlowHost, FlowLayout
        from .section import Section
        self._options_window = QDialog(self, Qt.Window)
        self._options_window.setWindowTitle(tr("Options"))
        self._options_window.setObjectName("OverviewOptionsWindow")
        self._options_window.setProperty("spacrSettingsPopup", True)
        self._options_window.setWindowModality(Qt.NonModal)
        self._options_window.resize(880, 680)
        options_layout = QVBoxLayout(self._options_window)
        content = QWidget(self._options_window)
        self._settings_content = content
        self._settings_host = None
        controls = QVBoxLayout(content)
        categories = {}
        for name in ("Source", "Figure settings", "Display", "Annotations", "Comparisons", "Export"):
            section = Section(name, expanded=True)
            controls.addWidget(section)
            categories[tr(name)] = section
        self._categories = categories
        essential = FlowHost(self)
        essential_flow = FlowLayout(essential)
        while column.count():
            item = column.takeAt(0)
            if item.widget() in (splitter, self.status):
                continue
            layout = item.layout()
            members = [layout.itemAt(index).widget() for index in range(layout.count())] if layout else [item.widget()]
            category = tr("Display")
            if self.source in members or self.regex in members:
                category = tr("Source")
            elif self.annotation_scope in members:
                category = tr("Annotations")
            elif any(widget in members for widget in (self.export_scope, self.grouping, self.use_selection, self.measurements)):
                category = tr("Export")
            if layout is not None:
                host = essential if layout in (self._navigation_layout, self._axes_layout) else FlowHost(content)
                flow = essential_flow if host is essential else FlowLayout(host)
                while layout.count():
                    child = layout.takeAt(0)
                    flow.addWidget(child.widget())
                layout.setParent(None)
                if host is not essential:
                    categories[category].add_widget(host)
            else:
                categories[category].add_widget(item.widget())
        controls.addStretch(1)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        scroll.setMinimumHeight(100)
        self._controls_scroll = scroll
        options_layout.addWidget(scroll)
        self.options_button = QPushButton(tr("Options"))
        self.options_button.setCheckable(True)
        self.options_button.toggled.connect(self._show_options)
        self._options_window.finished.connect(self._options_closed)
        essential_flow.addWidget(self.options_button)
        fit = QPushButton(tr("Fit plate"))
        fit.clicked.connect(lambda: self._zoom_plate())
        essential_flow.addWidget(fit)
        essential_scroll = QScrollArea(content)
        essential_scroll.setWidgetResizable(True)
        essential_scroll.setFrameShape(QScrollArea.NoFrame)
        essential_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        essential_scroll.viewport().setAutoFillBackground(False)
        essential_scroll.setWidget(essential)
        self._essential_scroll = essential_scroll
        self.installEventFilter(self)
        essential_scroll.viewport().installEventFilter(self)
        essential.installEventFilter(self)
        categories[tr("Figure settings")].add_widget(essential_scroll)
        categories[tr("Comparisons")].add_widget(self._build_comparison_controls())
        column.addWidget(splitter, 1)
        column.addWidget(self.status)

    def _show_options(self, on):
        """Open the separate controls window with normal keyboard focus."""
        if on and self._settings_host is not None and self._settings_host.widget() is self._settings_content:
            self._controls_scroll.setWidget(self._settings_host.takeWidget())
        self._options_window.setVisible(on)
        if on:
            self._options_window.raise_()
            self._options_window.activateWindow()

    def _options_closed(self, *_):
        """Keep the Options action consistent with its independent window."""
        if self._settings_host is not None and self._controls_scroll.widget() is self._settings_content:
            self._settings_host.setWidget(self._controls_scroll.takeWidget())
        self.options_button.setChecked(False)

    @staticmethod
    def _display_options_key(options):
        """Keep potentially large immutable advanced plans out of UI cache keys."""
        values = dict(options)
        advanced = values.pop("advanced_normalization", None)
        layers = tuple(id(values.pop(name)) if values.get(name) is not None else None
                       for name in ("channel_limits", "image_overrides"))
        return repr(values), id(advanced) if advanced is not None else None, layers

    def _axis_options(self):
        """Capture internal acquisition axes independently from filename identifiers."""
        time, z = self.time_axis.currentData(), self.z_axis.currentData()
        return dict(time_index=time if isinstance(time, int) else 0,
                    z_index=z if isinstance(z, int) else 0,
                    z_mode=self.z_mode.currentData() or "plane")

    def _selection_options(self, fields):
        """Retain each selected field or well's acquisition context across plates."""
        if self.export_scope.currentData() != "selection" and not self.use_selection.isChecked():
            return {}
        return {repr(field.identity): dict(self._selected_field_options.get(field.identity,
            self._selected_well_options.get((field.plate, field.well), self._axis_options())))
            for field in fields}

    def _display_kwargs(self, view=False):
        """Freeze selected stack axes and the immutable normalization plan.

        :param view: also carry the display-only enhancement chain, for the
            on-screen views; exports and measurements never receive it.
        """
        time = self.time_axis.currentData()
        z = self.z_axis.currentData()
        extra = {}
        if view:
            if self._enhancement is not None:
                extra["enhancement"] = self._enhancement
            if self._enhance_channels is not None:
                extra["enhancement_channels"] = self._enhance_channels
            if self._normalize_channels is not None:
                extra["percentile_channels"] = self._normalize_channels
            if self._channel_limits:
                extra["channel_limits"] = self._channel_limits
            if self._image_overrides:
                extra["image_overrides"] = self._image_overrides
        extra.update(self._channel_display_options(view))
        return dict(extra, normalization=self._normalization_plan,
                    normalization_context=self._normalization_context,
                    time_index=time if isinstance(time, int) else 0,
                    z_index=z if isinstance(z, int) else 0,
                    z_mode=self.z_mode.currentData() or "plane",
                    advanced_normalization=self._advanced_model if self.advanced.isChecked() else None,
                    channel_colors=self._display_colors(),
                    color_mode=self._color_mode)

    @staticmethod
    def _natural(value):
        """Sort acquisition identifiers by their numeric components.

        :param value: the literal acquisition identifier.
        :returns: comparable typed text and numeric tokens.
        """
        import re
        return tuple((1, int(part)) if part.isdigit() else (0, part.casefold())
                     for part in re.split(r"(\d+)", str(value)))

    def _populate_axes(self, fields, info=None):
        """Use literal metadata identifiers or declared stack indices, never captions.

        :param fields: immutable source records in this dataset.
        :param info: optional header-only channel and stack-axis discovery.
        """
        info = info or {}
        specs = ((self.fov_axis, "field_id", None), (self.time_axis, "time_id", "time_count"),
                 (self.z_axis, "z_id", "z_count"))
        for combo, attribute, count_name in specs:
            previous = combo.currentData()
            values = sorted({getattr(field, attribute, "") for field in fields
                             if getattr(field, attribute, "")}, key=self._natural)
            combo.blockSignals(True)
            combo.clear()
            if attribute == "field_id":
                combo.addItem(tr("All fields"), None)
            if values:
                for value in values:
                    combo.addItem(str(value), value)
            elif count_name:
                if info:
                    count = max((record.get(count_name, 1) for record in info.get("files", ())), default=1)
                    for value in range(count):
                        combo.addItem(str(value), value)
                else:
                    configured = self._overview_options.get(
                        "time_index" if attribute == "time_id" else "z_index", 0)
                    value = previous if isinstance(previous, int) else int(configured)
                    combo.addItem(str(value), value)
            if previous is None and attribute in ("time_id", "z_id"):
                previous = self._overview_options.get("time_index" if attribute == "time_id" else "z_index", 0)
            index = combo.findData(previous)
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.blockSignals(False)
        separate_z = any(getattr(field, "z_id", "") for field in fields)
        self.z_mode.setEnabled(not separate_z)

    def _prepare_display(self):
        """Resolve reference normalization and real axis headers off the GUI thread."""
        self._prepare_timer.stop()
        if not self._fields:
            return
        self._display_generation += 1
        self._normalization_ready = False
        if self._index_jobs.is_busy():
            self._display_pending = True
            return
        generation = self._display_generation
        fields, cancelled = self._fields, self._cancelled
        try:
            channels, limits = self._choices()
        except (ValueError, TypeError) as exc:
            self.status.setText(str(exc))
            return
        options = dict(self._overview_options.get("normalization") or {})
        options["percentiles"] = limits if limits is not None else (0., 100.)
        options["conditions"] = dict(self._conditions)
        captured = self._display_kwargs()
        axes = {key: captured[key] for key in ("time_index", "z_mode", "z_index")}
        advanced_enabled = self.advanced.isChecked()
        advanced_options = self._normalization_options()
        annotations = self._annotation_snapshot()
        time, z = self.time_axis.currentData(), self.z_axis.currentData()
        sampled_fields = tuple(field for field in fields
            if (not getattr(field, "time_id", "") or field.time_id == time)
            and (not getattr(field, "z_id", "") or field.z_id == z))
        if options.get("mode") == "reference_average" and not advanced_enabled:
            plate = self.plates.currentData()
            fov = self.fov_axis.currentData()
            representatives = {}
            for field in sampled_fields:
                if field.plate == plate and (fov is None or field.field_id == fov):
                    representatives.setdefault(field.well, []).append(field)
            sampled_fields = tuple(field for well in sorted(representatives, key=self._natural)
                for field in representatives[well][:self.representatives.value()])
        options["selection"] = dict(field_id=self.fov_axis.currentData(),
            filename_time=time if isinstance(time, str) else None,
            filename_z=z if isinstance(z, str) else None,
            time_index=axes["time_index"], z_mode=axes["z_mode"], z_index=axes["z_index"],
            representatives=self.representatives.value(), seed=self.seed.value(),
            generation=generation)
        self._normalization_plate = self.plates.currentData()
        self._normalization_ready = False
        def prepare():
            """Read metadata and bounded reference samples without Qt objects.

            :returns: header metadata and an immutable display plan.
            """
            info = _overview_channel_info(fields[0])
            plan, context, advanced = None, None, None
            if advanced_enabled:
                if advanced_options["scope"] == "image":
                    for channel in channels:
                        _overview_channel_normalization(advanced_options, channel)
                    advanced = dict(options=advanced_options)
                else:
                    advanced = _overview_advanced_normalization(sampled_fields, channels,
                        advanced_options, annotations, cancelled,
                        read_planes=lambda field, selected: _overview_raw_planes(field, selected, edge=256, **axes))
                advanced["selection"] = dict(axes)
                advanced.setdefault("provenance", {})["sampling_axes"] = (
                    "per_selected_image" if advanced_options["scope"] == "image" else dict(axes))
            elif self._overview_options.get("normalization"):
                plan, context = _overview_prepare_normalization(sampled_fields, channels, options,
                    lambda field, selected, edge=256: tuple(_overview_raw_planes(
                        field, selected, edge=edge, **axes)), cancelled)
            return info, plan, context, advanced
        def prepared(result):
            """Install only the still-current source generation.

            :param result: header metadata and the resolved normalization plan.
            """
            if (cancelled.is_set() or fields is not self._fields
                    or generation != self._display_generation):
                return
            info, self._normalization_plan, self._normalization_context, self._advanced_model = result
            self._populate_axes(fields, info)
            self._install_channel_buttons(info)
            self._normalization_ready = True
            self._request_visuals()
            self._reload_image_views()
            if self._individual:
                item = self.fields.currentItem()
                if item is not None:
                    self._field(item)
        self._index_jobs.submit(prepare, prepared)

    def _finish_display_prepare(self, *_):
        """Coalesce display-plan requests behind the current bounded worker.

        :param _: unused completion-signal arguments.
        """
        if self._display_pending and not self._cancelled.is_set():
            self._display_pending = False
            self._prepare_display()

    def _axis_changed(self, *_):
        """Retain the same well while changing an explicit acquisition axis.

        :param _: unused acquisition-control signal arguments.
        """
        if not self._fields:
            return
        individual = self._individual
        self._displayed_montage = None
        self._preview_edge = 256
        self._refresh_scope_limits()
        self._filter()
        if self._overview_options.get("normalization") or self.advanced.isChecked():
            self._prepare_display()
        if individual and self.fields.count():
            self.fields.setCurrentRow(0)
            self._field(self.fields.item(0))
        elif not self._visible:
            self.view.clear_image()
            self.status.setText(tr("No field exists for this well and acquisition-axis selection."))

    def _resolution_changed(self, *_):
        """Redraw the open views at the new image quality once the slider rests.

        Dragging the slider restarts a short timer, so only its resting
        position is decoded; every decode runs on a worker thread.
        """
        self._original_image = self._quality_original()
        self._quality_timer.start()

    def _apply_resolution(self):
        """Rebuild the plate, well and image views at the chosen quality.

        The plate thumbnails, every well view's tiles, every stacked image and
        the enlarged image load again at once, keeping each image's zoom; the
        top position loads the original pixels. Raw data and learned display
        limits are unchanged.
        """
        if self._controls_lost or self._cancelled.is_set():
            return
        self._original_image = self._quality_original()
        self._cache.clear()
        self._thumbnail_cache.clear()
        self._displayed_montage = None
        self._displayed_field = None
        self._preview_edge = self._quality_cap()
        if self._native_item is not None and self.view._item is not None:
            self.view.scene().removeItem(self._native_item)
            self._native_item = self._native_key = None
        self._request_visuals()
        self._reload_image_views()
        item = self.fields.currentItem()
        if self._individual and item is not None:
            self._field(item, preserve_view=True)

    def _wheel_mode_changed(self):
        """Apply the explicit wheel action without modifying any acquisition value."""
        self.view.wheel_mode = self.wheel_axis.currentData()

    def _step_axis(self, delta):
        """Step the selected field, Z or time axis within its real bounds.

        :param delta: one positive or negative wheel step.
        """
        combo = {"field": self.fov_axis, "z": self.z_axis,
                 "time": self.time_axis}.get(self.wheel_axis.currentData())
        if combo is not None:
            combo.setCurrentIndex(max(0, min(combo.count() - 1, combo.currentIndex() + delta)))

    def _request_detail(self):
        """Request a bounded finer source decode only after genuine user zoom."""
        if not self._individual or self.view._item is None or self._original_image:
            return
        width = self.view._item.pixmap().width()
        displayed = width * self.view.transform().m11()
        desired = min(4096, max(256, int(math.ceil(displayed * device_ratio(self.view.viewport())))))
        desired = min(self._quality_cap(),
            next((edge for edge in (256, 512, 1024, 2048, 4096) if edge >= desired), 4096))
        self._request_native_detail()
        if desired > self._preview_edge:
            self._preview_edge = desired
            item = self.fields.currentItem()
            if item is not None:
                self._field(item, preserve_view=True)

    def _request_native_detail(self, *_):
        """Debounce original-pixel crops after zooming or panning a field."""
        if self._individual and self.view._user_zoomed and not self._cancelled.is_set():
            self._native_timer.start()

    def _native_finished(self, *_):
        """Request the latest camera after an in-flight native crop settles."""
        if self._native_pending:
            self._native_pending = False
            self._request_native_detail()

    def _load_native_detail(self):
        """Overlay a bounded source-resolution crop in the retained preview scene.

        The crop is skipped while a source pixel covers less than half a
        device pixel, the same device-pixel measure :meth:`_request_detail`
        uses. On a 2x Retina screen a source pixel at a third of a logical
        pixel still spans two thirds of a physical one, so gating on logical
        pixels withheld the original detail there that a 1x screen would get.
        """
        if (not self._individual or self._cancelled.is_set() or self.view._item is None
                or self._original_image):
            return
        if self._native_jobs.is_busy():
            self._native_pending = True
            return
        item = self.fields.currentItem()
        if item is None:
            return
        field = item.data(Qt.UserRole)
        shape = self._source_shapes.get(field.identity)
        if not shape:
            return
        channels, limits = self._choices()
        if len(channels) > 3 or self._mosaic(channels):
            self.status.setText(tr("For original-resolution detail, toggle up to three channels or inspect one channel individually."))
            return
        preview = self.view._item.pixmap()
        height, width = shape
        scale_x, scale_y = preview.width() / width, preview.height() / height
        if self.view.transform().m11() * scale_x * device_ratio(self.view.viewport()) < .5:
            return
        visible = self.view.mapToScene(self.view.viewport().rect()).boundingRect()
        x0, y0 = max(0, int(visible.left() / scale_x)), max(0, int(visible.top() / scale_y))
        x1, y1 = min(width, math.ceil(visible.right() / scale_x)), min(height, math.ceil(visible.bottom() / scale_y))
        if x1 <= x0 or y1 <= y0:
            return
        if x1 - x0 > 2048:
            centre = (x0 + x1) // 2
            x0, x1 = max(0, centre - 1024), min(width, centre + 1024)
        if y1 - y0 > 2048:
            centre = (y0 + y1) // 2
            y0, y1 = max(0, centre - 1024), min(height, centre + 1024)
        camera = (self.view.transform(), self.view.horizontalScrollBar().value(),
                  self.view.verticalScrollBar().value(), self.view.viewport().size(), preview.size())
        options, signature, cancelled = self._display_kwargs(view=True), self._visual_context(), self._cancelled
        region = y0, y1, x0, x1
        key = field.identity, signature, region, camera
        if self._native_key == key:
            return
        self._native_pending = False
        def done(pixels):
            """Publish only the current field, axes, model and camera crop."""
            current = self.fields.currentItem()
            if (cancelled.is_set() or not self._individual or current is None
                    or current.data(Qt.UserRole).identity != field.identity
                    or signature != self._visual_context() or self.view._item is None):
                return
            current_camera = (self.view.transform(), self.view.horizontalScrollBar().value(),
                self.view.verticalScrollBar().value(), self.view.viewport().size(), self.view._item.pixmap().size())
            if camera != current_camera:
                return
            from PySide6.QtGui import QTransform
            if self._native_item is not None:
                self.view.scene().removeItem(self._native_item)
            image = QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0], QImage.Format_RGB888).copy()
            self._native_item = self.view.scene().addPixmap(QPixmap.fromImage(image))
            self._native_item.setTransform(QTransform.fromScale(scale_x, scale_y))
            self._native_item.setPos(x0 * scale_x, y0 * scale_y)
            self._native_item.setZValue(1)
            self._native_key = key
        self._native_jobs.submit(lambda: _overview_pixels(field, channels, limits, edge=None,
            region=region, cancelled=cancelled, **options), done)

    def _representatives_changed(self):
        """Invalidate plate representatives while retaining source selections."""
        self._thumbnail_cache.clear()
        self._filter()
        if self._overview_options.get("normalization"):
            self._prepare_display()

    def _annotate(self):
        """Associate any metadata key with stable physical well identities."""
        plate = self.plates.currentData()
        scope, text = self.annotation_scope.currentData(), self.annotation_text.text().strip()
        key = self._annotation_name()
        if scope == "wells":
            addresses = set(self._selected_wells)
        else:
            row, column = self.rows.currentData(), self.columns.currentData()
            addresses = {(field.plate, field.well) for field in self._fields if field.plate == plate
                and (scope != "row" or row is not None and field.row == row)
                and (scope != "column" or column is not None and field.column == column)}
        self._apply_annotation(addresses, key, text)

    def _apply_annotation(self, addresses, key, text):
        """Set or remove one metadata value on plate-qualified wells.

        :param addresses: ``(plate, well)`` identities to annotate.
        :param key: the metadata key.
        :param text: the value; empty text removes the key.
        """
        if not addresses or not key:
            self.status.setText(tr("Choose a row, column or well selection before annotating."))
            return
        from ...measure import _image_annotation_column_name
        try:
            _image_annotation_column_name(key)
        except ValueError:
            self.status.setText(tr(
                "Annotation names use letters, digits and underscores, start with a letter and cannot name an image identity: {name}",
                name=key))
            return
        for address in addresses:
            if text:
                self._annotations.setdefault(address, {})[key] = text
                if key == "condition":
                    self._conditions[address] = text
            else:
                self._annotations.get(address, {}).pop(key, None)
                if key == "condition":
                    self._conditions.pop(address, None)
        self._fill_group_choices(self.grouping)
        self._fill_group_choices(self.normalization_scope, image=True)
        self._refresh_comparison_choices()
        self._persist_metadata()
        self._store_image_annotation(addresses, key, text)
        self._filter()
        if self._overview_options.get("normalization") or self.advanced.isChecked():
            self._prepare_display()

    def _store_image_annotation(self, addresses, key, text):
        """Save one annotation for every image of the wells in measurements.db.

        :param addresses: ``(plate, well)`` identities that were annotated.
        :param key: the validated annotation column.
        :param text: the value; empty text clears it.
        """
        from ...measure import _image_annotation_database, _set_image_annotations

        source = self._metadata_source
        if not source:
            return
        path = _image_annotation_database(source)
        wells = sorted(addresses)
        fields = tuple(field for field in self._fields if (field.plate, field.well) in addresses)
        with self._annotation_write_lock:
            self._image_annotation_queue.append((path, fields, key, text, source, wells,
                                                 self._annotation_file_actions))
            if self._image_annotation_running:
                return
            self._image_annotation_running = True
        def write():
            """Write queued annotations one at a time, in the order they were made.

            Two annotations in a row used to be written by two concurrent jobs,
            so the earlier one could land last in the database and both could
            contend for its write lock.

            :returns: ``(entry, count)`` per write; ``count`` is ``None`` after a refusal.
            """
            import sqlite3
            results = []
            try:
                while True:
                    with self._annotation_write_lock:
                        if not self._image_annotation_queue:
                            self._image_annotation_running = False
                            return results
                        entry = self._image_annotation_queue.pop(0)
                    target, chosen, name, value, origin, addressed, _actions = entry
                    try:
                        count = _set_image_annotations(target, chosen, name, value, source=origin,
                                                       wells=addressed)["count"]
                    except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
                        LOG.warning("image annotation not saved: %s", error)
                        count = None
                    results.append((entry, count))
            except BaseException:
                with self._annotation_write_lock:
                    self._image_annotation_running = False
                raise
        def written(results):
            """Report the last save, unless an annotation file was saved or loaded since.

            That later action's own report, such as a refused import, stays shown.

            :param results: ``(entry, count)`` pairs in write order.
            """
            if not results:
                return
            (target, _chosen, name, _value, _origin, _addressed, actions), count = results[-1]
            if self._annotation_file_actions != actions:
                return
            if count is None:
                self.status.setText(tr("The annotation could not be saved to {path}.", path=target))
            else:
                self.status.setText(tr("Saved {name} for {count} images in {path} (table image_annotations).",
                                       name=name, count=count, path=target))
        self._metadata_jobs.submit(write, written)

    def _compare(self):
        """Render only explicitly selected fields, retaining their source identities."""
        self._comparison_request()
        fields = tuple(self._selected_fields.values())
        if not fields:
            self.status.setText(tr("Select fields by clicking or dragging a selection rectangle."))
            return
        if len(fields) > 24:
            self.status.setText(tr("Compare up to 24 fields at a time."))
            return
        self._compare_fields(fields)

    def _compare_fields(self, fields, message=None):
        """Show up to 24 fields side by side in the image level.

        :param fields: the source fields, in display order.
        :param message: status text replacing the default preview count.
        """
        try:
            channels, limits = self._choices()
        except (ValueError, TypeError) as exc:
            self.status.setText(str(exc))
            return
        options, cancelled = self._display_kwargs(view=True), self._cancelled
        request = getattr(self, "_comparison_requests", 0)
        labels = self.metadata_toggle.isChecked()
        self._individual = False
        self._show_image_panel()
        signature = tuple(field.identity for field in fields), channels, limits, self._display_options_key(options)
        self._comparison_signature = signature
        def done(result):
            """Show only a comparison that still owns the current navigation.

            :param result: detached montage pixels and per-field errors.
            """
            if (not cancelled.is_set() and not self._individual
                    and self._comparison_signature == signature
                    and request == getattr(self, "_comparison_requests", 0)):
                pixels, errors = result
                if pixels is not None:
                    self._show_pixels(pixels)
                    self.status.setText(message or tr("{count} fields; {errors} unreadable previews.",
                                                      count=len(fields), errors=len(errors)))
        self._montage_jobs.submit(lambda: _overview_montage(
            fields, channels, limits, cancelled, labels=labels, **options), done)

    def _choices(self):
        """Validate one display/export configuration on the GUI thread."""
        try:
            channels = tuple(int(token.strip()) for token in self.channels.text().split(","))
        except (ValueError, TypeError):
            raise _OverviewValidationError(tr("Choose one or more nonnegative channel indices.")) from None
        if not channels or min(channels) < 0:
            raise _OverviewValidationError(tr("Choose one or more nonnegative channel indices."))
        low, high = self.low.value(), self.high.value()
        if self._editing_channel is not None and "percentile_low" in self._global_display:
            low, high = self._global_display["percentile_low"], self._global_display["percentile_high"]
        limits = (low, high) if self.normalise.isChecked() else None
        if limits is not None and limits[0] >= limits[1]:
            raise _OverviewValidationError(tr("The lower percentile must be below the upper percentile."))
        return channels, limits

    def _browse(self):
        """Ask for a source without touching the filesystem on the GUI thread."""
        from PySide6.QtWidgets import QFileDialog
        value = QFileDialog.getExistingDirectory(self, tr("Image source"))
        if value:
            self.source.setText(value)
            self._scan()

    def _metadata(self):
        """Reuse the import workbench to preview and accept a filename pattern."""
        from .import_workbench import ImportWorkbenchDialog
        from itertools import islice

        filenames = tuple(islice((path for field in self._fields for _, path in field.files), 500))
        if not filenames:
            self.status.setText(tr("Load the image index before opening filename metadata."))
            return
        self._metadata_dialog = ImportWorkbenchDialog(filenames, self.regex.text(), self)
        self._metadata_dialog.accepted.connect(self._accept_metadata)
        self._metadata_dialog.open()

    def _accept_metadata(self):
        """Apply the workbench's complete filename pattern without moving images."""
        self.regex.setText(self._metadata_dialog.workbench.regex.text())
        self._scan()

    def _cancel(self):
        """Cancel queued results and stop work at the next field boundary."""
        self._cancelled.set()
        self._comparison_signature = None
        self.status.setText(tr("Cancellation requested; the current image read may finish."))
        self._index_jobs.cancel()
        self._measurement_cancelled.set()
        self._native_timer.stop()
        self._native_jobs.cancel()

    def _scan(self):
        """Replace the source generation; old worker results cannot leak through."""
        import threading
        self._cancel()
        self._cancelled = threading.Event()
        self._cache.clear()
        self._thumbnail_cache.clear()
        self._displayed_montage = None
        self._visual_signature = None
        self._readout_planes = self._readout_loading = None
        self._clear_pixel_readout()
        self._individual = False
        self._selected_fields.clear()
        self._selected_wells.clear()
        self._selected_well_axes.clear()
        self._selected_well_options.clear()
        self._selected_field_options.clear()
        self._fields, self._visible = (), ()
        self._all_fields = ()
        for table in self._plate_table_list():
            table.clear()
        self.fields.clear()
        self.view.clear_image()
        source, regex, convention, cancelled = self.source.text(), self.regex.text(), self._convention, self._cancelled
        self.status.setText(tr("Indexing image metadata…"))
        source_scope = self._overview_options.get("source_scope", "recursive")
        well_layout = self._overview_options.get("well_layout")
        use_subfolders = tuple(self._overview_options.get("use_subfolders") or ())
        excluded = tuple(self._overview_options.get("excluded_plates") or ())
        patterns = dict(self._overview_options.get("plate_regex") or {})
        def read():
            """Reject unresolved wells on every browser load, retaining honest previews."""
            fields = _overview_matched_fields(_overview_index_sources(
                source, regex, convention, cancelled, source_scope=source_scope,
                well_layout=well_layout, use_subfolders=use_subfolders, excluded=excluded,
                patterns=patterns)[0], regex, patterns)
            if any(not field.well or field.unresolved for field in fields):
                raise _OverviewValidationError(tr("The filename regex must include a valid wellID for every image."))
            return fields
        self._index_jobs.submit(read, self._indexed)

    def _indexed(self, fields):
        """Show actual plate identities and explicitly retained unresolved fields.

        :param fields: immutable source records returned by the metadata indexer.
        """
        self._all_fields = tuple(fields)
        fields = _overview_limited_fields(fields, *self._load_limits)
        self._fields = fields
        self._source_paths = _overview_source_paths(self.source.text())
        self._metadata_source = self._source_paths[0] if self._source_paths else ""
        self._restore_metadata()
        self._restore_view_state()
        self._restore_comparison_groups()
        self._populate_axes(fields)
        channel_ids = sorted({channel for field in fields for channel, _ in field.files if channel},
                             key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))
        if channel_ids and self.channels.text() == "0":
            self.channels.setText(",".join(channel_ids[:3]))
        self.plates.blockSignals(True)
        shown = self.plates.currentData()
        self.plates.clear()
        names = self._overview_options.get("plate_names") or {}
        for plate in sorted({field.plate for field in fields}, key=_overview_natural_key):
            self.plates.addItem(str(names.get(plate) or plate), plate)
        if shown is not None and self.plates.findData(shown) >= 0:
            self.plates.setCurrentIndex(self.plates.findData(shown))
        options = self._overview_options.get("normalization") or {}
        reference = options.get("reference")
        if (options.get("mode") == "reference_average"
                and isinstance(reference, (tuple, list)) and len(reference) == 2):
            index = self.plates.findData(reference[0])
            if index >= 0:
                self.plates.setCurrentIndex(index)
        self.plates.blockSignals(False)
        self._sync_plate_buttons()
        self._back()
        if not self._geometry_valid:
            return
        self._prepare_display_after_paint()
        self._refresh_scope_limits()
        self.status.setText(tr("{count} fields; {unresolved} with unresolved metadata.",
                               count=len(fields), unresolved=sum(f.unresolved for f in fields)))

    def _back(self):
        """Return to the current plate, retaining its identity and filters."""
        self._scope = (self.plates.currentData(), None)
        self._individual = False
        self._comparison_signature = None
        for panel in tuple(self._well_panels):
            self._close_well_panel(panel, refresh=False)
        self.fields = self._well_panel.field_list
        for panel in tuple(self._image_views):
            self._close_image_view(panel)
        self._image_panel.hide()
        self._image_column.hide()
        self.wells.show()
        self._plate_fit = True
        self._plate_cell_extent = None
        self.view.clear_image()
        self._page = 0
        self._filter()

    @staticmethod
    def _checked_plate_geometry(fields, geometry):
        """Refuse an explicit format that cannot contain the known source wells.

        :param fields: source records for this physical plate.
        :param geometry: explicitly chosen row and column counts, or None.
        :returns: the validated dimensions, or None for metadata-derived geometry.
        """
        if geometry is None:
            return None
        rows, columns = int(geometry[0]), int(geometry[1])
        if rows < 1 or columns < 1 or any(not field.unresolved and
                (int(field.row) > rows or field.column > columns) for field in fields):
            raise _OverviewValidationError(tr(
                "The selected geometry does not contain the indexed wells ({rows} rows, {columns} columns required).",
                rows=max((int(field.row) for field in fields if not field.unresolved), default=0),
                columns=max((field.column for field in fields if not field.unresolved), default=0)))
        return rows, columns

    def _filter(self, *_):
        """Filter metadata and show bounded field pages without image reads.

        Every shown plate (:meth:`_shown_plates`) gets its own table; the
        well maps are keyed by ``(plate, well)``, so the same well on two
        plates never meets.

        :param _: unused plate, row or column selection arguments.
        """
        current_identities = {id(fields): fields.currentItem().data(Qt.UserRole).identity
                              for fields in self._field_lists() if fields.currentItem() is not None}
        self._sync_plate_tables()
        plate = self.plates.currentData()
        self._comparison_signature = None
        if self._scope[0] != plate:
            self._scope = (plate, None)
            for panel in tuple(self._well_panels):
                self._close_well_panel(panel, refresh=False)
            self.fields = self._well_panel.field_list
        available = tuple(f for f in self._fields if f.plate == plate)
        try:
            geometry = self._checked_plate_geometry(available, self._overview_options.get("geometry"))
        except _OverviewValidationError as error:
            self._geometry_error = str(error)
            self._geometry_valid = False
            self._normalization_ready = False
            self._visible = ()
            for table in self._plate_tables.values():
                self._clear_plate_table(table)
            self.fields.clear()
            self.view.clear_image()
            self.status.setText(str(error))
            return
        self._geometry_valid = True
        self._geometry_error = None
        lines = tuple(f for f in self._fields if f.plate in self._plate_tables) or available
        for combo, values, title in ((self.rows, sorted({f.row for f in lines if f.row}, key=int), tr("All rows")),
                                     (self.columns, sorted({f.column for f in lines if f.column}), tr("All columns"))):
            old = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(title, None)
            for value in values:
                combo.addItem(str(value), value)
            if old in values:
                combo.setCurrentIndex(combo.findData(old))
            combo.blockSignals(False)
        row, col = self.rows.currentData(), self.columns.currentData()
        axes = tuple(combo.currentData() for combo in (self.fov_axis, self.time_axis, self.z_axis))
        self._well_representatives, self._well_fields, self._well_positions = {}, {}, {}
        self._plate_available = {}
        self._installed_thumbnails.clear()
        self._await_plate_paint()
        for shown, table in self._plate_tables.items():
            plate_geometry = geometry
            if shown != plate:
                try:
                    plate_geometry = self._checked_plate_geometry(
                        tuple(f for f in self._fields if f.plate == shown), self._overview_options.get("geometry"))
                except _OverviewValidationError as error:
                    self._clear_plate_table(table)
                    self._plate_blocks[table][1].setToolTip(str(error))
                    continue
            filled = self._fill_plate_table(table, shown, row, col, axes, plate_geometry)
            if filled is None and shown == plate:
                self.status.setText(tr("Plate coordinates exceed the overview grid limit."))
                return
            if filled is not None:
                self._plate_available[shown] = filled
        available = self._plate_available.get(plate, ())
        _fov, time, z = axes
        for address in self._selected_wells:
            if address[0] in self._plate_tables:
                self._selected_well_axes[address] = (time, z)
        self._visible = tuple(f for f in available if self._scope[1] is None or f.well == self._scope[1])
        self._page = min(self._page, max(0, (len(self._visible) - 1) // 24))
        self.fields._overview_page = self._page
        for fields in self._field_lists():
            records = (self._visible if fields is self.fields
                       else tuple(self._well_fields.get((plate, fields._overview_well), ())))
            page = min(fields._overview_page, max(0, (len(records) - 1) // 24))
            self._fill_field_list(fields, records[page * 24:(page + 1) * 24],
                                  current_identities.get(id(fields)))
        if not self._visible:
            self.view.clear_image()
        self._label_plate_blocks()
        if self._plate_fit or self._plate_cell_extent is None:
            self._zoom_plate()
        else:
            self._apply_plate_extent()
        if (self._overview_options.get("normalization")
                and self._normalization_plate != plate):
            self._prepare_display()
        self._request_visuals()

    def _fill_plate_table(self, table, plate, row, col, axes, geometry):
        """Lay out one plate's wells in ``table``, filed under ``(plate, well)``.

        Wells beyond Overview max wells are drawn hatched as not loaded, per plate.

        :param table: the plate's table.
        :param plate: the plate's session name.
        :param row: the row filter, or ``None`` for every row.
        :param col: the column filter, or ``None`` for every column.
        :param axes: the shown ``(field, time, z)``.
        :param geometry: the plate's checked ``(rows, columns)``, or ``None``.
        :returns: the plate's fields at the shown axes, or ``None`` when its
            coordinates exceed the grid limit.
        """
        from collections import Counter
        from .sortable_table import table_item
        from ...schema import letters_from_row_index

        table._overview_plate = plate
        available = tuple(f for f in self._fields if f.plate == plate and (row is None or f.row == row)
                          and (col is None or f.column == col))
        known_wells = {field.well for field in available}
        loaded_wells = {field.well for field in self._fields if field.plate == plate}
        skipped = {f.well: (int(f.row), f.column) for f in self._all_fields
                   if f.plate == plate and not f.unresolved and f.well and f.well not in loaded_wells
                   and (row is None or f.row == row) and (col is None or f.column == col)}
        addresses = {f.well: (int(f.row), f.column) for f in available if not f.unresolved}
        addresses.update({well: position for well, position in skipped.items() if well not in addresses})
        rows = max((position[0] for position in addresses.values()), default=0)
        columns = max((position[1] for position in addresses.values()), default=1)
        if geometry is not None:
            rows, columns = int(geometry[0]), int(geometry[1])
        if rows > 512 or columns > 512:
            self._clear_plate_table(table)
            return None
        fov, time, z = axes
        available = tuple(field for field in available
            if (fov is None or getattr(field, "field_id", "") == fov)
            and (not getattr(field, "time_id", "") or field.time_id == time)
            and (not getattr(field, "z_id", "") or field.z_id == z))
        counts = Counter(f.well for f in available)
        for field in available:
            self._well_fields.setdefault((plate, field.well), []).append(field)
            self._well_representatives.setdefault((plate, field.well), field)
        table.blockSignals(True)
        table.clear()
        coverslip = geometry == (1, 1) and not addresses and known_wells == {""}
        table.setRowCount(1 if coverslip else rows + int("" in known_wells))
        table.setColumnCount(columns)
        table.setHorizontalHeaderLabels([str(n) for n in range(1, columns + 1)])
        table.setVerticalHeaderLabels([tr("Unresolved")] if coverslip else
            [letters_from_row_index(n) for n in range(1, rows + 1)]
            + ([tr("Unresolved")] if "" in known_wells else []))
        snapshot = self._annotation_snapshot()
        language = current_language()
        unresolved = tr("Unresolved")
        no_image = tr("No image at the selected acquisition axes.")
        shown = self.representatives.value()
        for well in sorted(known_wells):
            count = counts[well]
            r, c = addresses.get(well, (1 if coverslip else rows + 1, 1))
            item = table_item(tr("{well}\n{count} fields", language=language, well=well or unresolved, count=count))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            item.setData(Qt.UserRole, well)
            detail = [item.text(), self._conditions.get((plate, well), "")]
            detail.extend(str(key) + ": " + str(value)
                          for key, value in snapshot.get((plate, well), {}).items())
            detail.extend(field.caption for field in self._well_fields.get((plate, well), ())[:shown])
            if not count:
                detail.append(no_image)
            item.setToolTip("\n".join(part for part in detail if part))
            table.setItem(r - 1, c - 1, item)
            item.setSelected((plate, well) in self._selected_wells)
            self._well_positions[(plate, well)] = (r - 1, c - 1)
            table.setRowHeight(r - 1, 150)
        for well, (r, c) in sorted(skipped.items()):
            if well in known_wells or r > table.rowCount() or c > table.columnCount():
                continue
            item = table_item(tr("{well}\nNot loaded", language=language, well=well))
            item.setFlags(Qt.NoItemFlags)
            item.setData(_OVERVIEW_NOT_LOADED_ROLE, True)
            item.setToolTip(tr("{well} is not loaded: it is beyond Overview max wells.", well=well))
            table.setItem(r - 1, c - 1, item)
        for index in range(columns):
            table.setColumnWidth(index, 180)
        table.blockSignals(False)
        return available

    def _remember_selection(self):
        """Retain explicit field selections across pages, well views and plates."""
        for item in (fields.item(index) for fields in self._field_lists()
                     for index in range(fields.count())):
            field = item.data(Qt.UserRole)
            if item.isSelected():
                self._selected_fields[field.identity] = field
                self._selected_field_options.setdefault(field.identity, self._axis_options())
            else:
                self._selected_fields.pop(field.identity, None)
                self._selected_field_options.pop(field.identity, None)

    def _clear_selection(self):
        """Clear both visible and retained explicit selections."""
        self._selected_fields.clear()
        self._selected_wells.clear()
        self._selected_well_axes.clear()
        self._selected_well_options.clear()
        self._selected_field_options.clear()
        for fields in self._field_lists():
            fields.clearSelection()
        for table in self._plate_table_list():
            table.clearSelection()

    def _well(self, item):
        """Zoom into one well while retaining the current plate and filters.

        :param item: the selected physical well cell.
        """
        self._focus_plate_table(item.tableWidget())
        self._open_well_panel(item.data(Qt.UserRole))
        self._scope = (self.plates.currentData(), item.data(Qt.UserRole))
        self._comparison_signature = None
        self._show_image_panel()
        self._individual = False
        self._page = 0
        self._filter()

    def _turn(self, delta):
        """Move one bounded field page at a time.

        :param delta: the requested positive or negative page step.
        """
        self._individual = False
        self._comparison_signature = None
        self._page = max(0, min((len(self._visible) - 1) // 24, self._page + delta))
        self._filter()

    def _refresh_image(self, *_):
        """Re-render only the current field after display controls change.

        :param _: unused display-control signal arguments.
        """
        if self._controls_lost:
            return
        self._refresh_scope_limits()
        self._displayed_montage = None
        self._comparison_signature = None
        self._reload_image_views()
        if self._overview_options.get("normalization") or self.advanced.isChecked():
            self._prepare_display()
            return
        self._request_visuals()
        item = self.fields.currentItem()
        if item is not None and self._individual:
            self._field(item)

    def _current_field_changed(self, current, previous):
        """Follow keyboard navigation while ignoring list resets during filtering."""
        if current is not None:
            self._field(current)

    def _field(self, item, preserve_view=False):
        """Progressively decode one field without using Qt objects on a worker.

        :param item: the selected field's list item.
        :param preserve_view: retain the current source-relative zoom and center.
        """
        try:
            channels, limits = self._choices()
        except (ValueError, TypeError) as exc:
            self.status.setText(str(exc))
            return
        if self._cancelled.is_set() or not self._geometry_valid:
            return
        field = item.data(Qt.UserRole)
        self._show_image_panel()
        self._individual = True
        self._main_view_field = field
        self._refresh_adjusted_badges()
        self._update_image_metadata(field)
        self._comparison_signature = None
        self._displayed_montage = None
        if not self._normalization_ready:
            self.view.clear_image()
            return
        options = self._display_kwargs(view=True)
        plain_options, heavy = options, self._original_chain(options) if self._original_image else None
        if heavy is not None:
            options = dict(options, enhancement=heavy)
        source_key = (field.identity, channels, limits, self._display_options_key(options),
                      self._original_generation if heavy is not None else None)
        if source_key != self._displayed_field:
            same_image = (self._displayed_field is not None and self._displayed_field[:2] == source_key[:2]
                          and self.view._item is not None)
            if same_image:
                self._preview_edge = min(self._preview_edge, self._quality_cap())
                preserve_view = True
            else:
                self._preview_edge = min(256, self._quality_cap())
                preserve_view = False
            self._displayed_field = source_key
        edge = None if self._original_image else self._preview_edge
        key = source_key + (edge,)
        if not preserve_view:
            self.view.clear_image()
        if self._image_jobs.is_busy():
            self._image_pending = True
            return
        self._image_pending = False
        if key in self._cache:
            self._cache.move_to_end(key)
            self._show_pixels(self._cache[key], preserve_view)
            return
        cancelled, signature = self._cancelled, self._visual_context()
        target_edge = self._quality_cap()
        def done(result):
            """Cache bounded pixels and accept only the exact current axis/view.

            :param result: detached pixels, the original header dimensions and
                any reason the full-resolution steps were not applied.
            """
            pixels, shape, note = result
            if cancelled.is_set():
                return
            if heavy is not None:
                self.status.setText(note or tr("Original image shown with {steps}.",
                                               steps=_overview_heavy_steps(heavy)))
            self._cache[key] = pixels
            self._source_shapes[field.identity] = shape
            while len(self._cache) > 8:
                self._cache.popitem(last=False)
            current = self.fields.currentItem()
            if (self._individual and current is not None
                    and current.data(Qt.UserRole).identity == field.identity
                    and self._visual_context() == signature
                    and self._displayed_field == source_key):
                self._show_pixels(pixels, preserve_view)
                if edge is not None and edge == min(256, target_edge) and edge < target_edge:
                    self._preview_edge = target_edge
                    self._image_pending = True
        def read():
            """Read original dimensions and a bounded preview without Qt access."""
            info = _overview_channel_info(field)
            record = info["files"][0]
            shape = tuple(record["shape"][record["axes"].index(axis)] for axis in "YX")
            if heavy is None:
                return _overview_pixels(field, channels, limits, edge=edge, **options), shape, ""
            try:
                return _overview_pixels(field, channels, limits, edge=edge, **options), shape, ""
            except Exception as error:
                note = tr("Original image shown without {steps}: {error}",
                          steps=_overview_heavy_steps(heavy), error=error)
                return _overview_pixels(field, channels, limits, edge=edge, **plain_options), shape, note
        if heavy is not None:
            self.status.setText(tr("Applying {steps} to the original image…",
                                   steps=_overview_heavy_steps(heavy)))
        self._image_jobs.submit(read, done)

    def eventFilter(self, watched, event):
        """Request only newly visible well images after viewport paints or resizes.

        :param watched: the actual plate or field-list viewport.
        :param event: a paint or resize event that may expose image tiles.
        :returns: the inherited event-filter result.
        """
        if (event.type() in (QEvent.FontChange, QEvent.StyleChange, QEvent.Polish,
                             QEvent.LayoutRequest, QEvent.Resize, QEvent.HideToParent,
                             QEvent.ParentChange)
                and (watched is getattr(self, "_module_actions", None)
                     or (event.type() != QEvent.LayoutRequest and watched is
                         getattr(getattr(self, "_module_actions", None), "viewport", lambda: None)())
                     or watched is getattr(self, "_module_action_content", None)
                     or watched is self
                     or watched is getattr(getattr(self, "_essential_scroll", None), "viewport", lambda: None)()
                     or watched is getattr(getattr(self, "_essential_scroll", None), "widget", lambda: None)()
                     or (event.type() == QEvent.Resize and watched in
                         getattr(self, "_allocation_viewports", ()))
                     or watched in getattr(self, "_module_action_buttons", ()))
                and not getattr(self, "_module_actions_retired", True)
                and not getattr(self, "_fitting_module_actions", False)):
            self._module_action_fit_timer.start(0)
        if event.type() == QEvent.NativeGesture:
            owner = getattr(self, "_readout_viewports", {}).get(watched)
            if isinstance(owner, _OverviewImageView) and _overview_pinch(owner, event):
                return True
        if event.type() in (QEvent.MouseMove, QEvent.Leave):
            owner = getattr(self, "_readout_viewports", {}).get(watched)
            if owner is not None and getattr(self, "_pixel_readout", None) is not None:
                self._readout_event(owner, event)
        header = getattr(watched, "_overview_header", None)
        if header is not None and event.type() == QEvent.MouseButtonPress:
            header_table, axis = header
            self._header_press_lines = (axis, {index.row() if axis == "row" else index.column()
                                               for index in header_table.selectedIndexes()})
        table = self._plate_table_at(watched)
        if table is not None:
            if event.type() == QEvent.MouseButtonPress:
                self._drag_start = event.position().toPoint()
                self._drag_selected = False
                self._press_modifiers = event.modifiers()
                self._press_plate(table, event)
            elif event.type() == QEvent.MouseMove and self._drag_start is not None and event.buttons() & Qt.LeftButton:
                from PySide6.QtWidgets import QApplication
                self._drag_selected |= (event.position().toPoint() - self._drag_start).manhattanLength() >= QApplication.startDragDistance()
            elif event.type() == QEvent.MouseButtonRelease:
                self._drag_start = None
        if (table is not None and event.type() == QEvent.Wheel
                and event.angleDelta().y()):
            position = event.position().toPoint()
            old_width, old_height = self._plate_extents.get(table, self._plate_cell_extent) or (1, 1)
            column = (table.horizontalScrollBar().value() + position.x()) / old_width
            row = (table.verticalScrollBar().value() + position.y()) / old_height
            self._zoom_plate(1 if event.angleDelta().y() > 0 else -1)
            table.doItemsLayout()
            width, height = self._plate_cell_extent
            table.horizontalScrollBar().setValue(round(column * width - position.x()))
            table.verticalScrollBar().setValue(round(row * height - position.y()))
            event.accept()
            return True
        if ((table is not None or watched in self._plate_table_list()) and event.type() == QEvent.KeyPress
                and event.key() in (Qt.Key_Return, Qt.Key_Enter)):
            self._focus_plate_table(table if table is not None else watched)
            self._open_selected_well()
            return True
        if table is not None and event.type() == QEvent.Resize and self._plate_fit:
            self._zoom_plate()
        if table is not None and event.type() == QEvent.Paint:
            self._plate_painted = True
        if ((table is not None or hasattr(watched.parentWidget(), "_overview_well"))
                and event.type() in (QEvent.Paint, QEvent.Resize)):
            self._request_visuals()
        return super().eventFilter(watched, event)

    def _plate_table_at(self, watched):
        """The shown plate table whose viewport is ``watched``, or ``None``."""
        return next((table for table in self._plate_table_list() if table.viewport() is watched), None)

    def _press_plate(self, table, event):
        """A plain left press on one plate starts a new selection: the other plates' selections clear.

        Ctrl or Shift extend the selection across plates, and a press on a
        selected well keeps the selection so it can be dragged.
        """
        if (event.button() != Qt.LeftButton or event.modifiers() & (Qt.ControlModifier | Qt.ShiftModifier)
                or len(self._plate_tables) < 2):
            return
        item = table.itemAt(event.position().toPoint())
        if item is not None and item.isSelected():
            return
        for other in self._plate_tables.values():
            if other is not table:
                other.clearSelection()

    def showEvent(self, event):
        """Resume bounded image requests when the retained dialog is shown."""
        super().showEvent(event)
        if self._cancelled.is_set():
            import threading
            self._cancelled = threading.Event()
            self._image_pending = False
            self._display_pending = False
            if self._fields and (not self._normalization_ready
                                 or self._normalization_context is not None):
                self._prepare_display()
        self._request_visuals()

    def _request_visuals(self, *_):
        """Coalesce viewport and navigation updates on one owned timer."""
        if self.isVisible() and not self._cancelled.is_set() and not self._controls_lost:
            self._visual_timer.start(0)

    def _visual_context(self):
        """Snapshot live navigation and display choices for stale-result checks."""
        try:
            channels, limits = self._choices()
        except (ValueError, TypeError):
            return None
        return (self._scope, self.rows.currentData(), self.columns.currentData(),
                self._page, self._display_generation,
                self.fov_axis.currentData(), self.time_axis.currentData(),
                self.z_axis.currentData(), self.z_mode.currentData(), channels, limits,
                self.preview_resolution.currentData(), self._enhancement)

    def _update_visuals(self):
        """Keep the plate's thumbnails current, then the open well's montage.

        The plate keeps one image per well on show whatever is open below it.
        """
        if (not self.isVisible() or self._cancelled.is_set() or self._controls_lost
                or not self._normalization_ready or not self._geometry_valid):
            return
        try:
            channels, limits = self._choices()
        except (ValueError, TypeError) as exc:
            self.status.setText(str(exc))
            return
        page = self._visible[self._page * 24:(self._page + 1) * 24]
        signature = self._visual_context()
        self._visual_signature = signature
        cancelled = self._cancelled
        options = self._display_kwargs(view=True)
        resolution = self._quality_cap()
        self._update_field_thumbnails(signature, channels, limits, options)
        self._update_plate_thumbnails(channels, limits, options, resolution, cancelled)
        if self._comparison_signature is not None:
            return
        if self._scope[1] is not None and not self._individual and page:
            labels = self.metadata_toggle.isChecked()
            if ((self._displayed_montage == signature and self._montage_labels == labels)
                    or self._montage_jobs.is_busy()):
                return
            def show_montage(result):
                """Display a montage only while its navigation choices match.

                :param result: montage pixels and read errors from the worker.
                """
                pixels, errors = result
                if (cancelled.is_set() or self._visual_context() != signature or self._individual
                        or pixels is None):
                    return
                self._show_pixels(pixels)
                self._displayed_montage, self._montage_labels = signature, labels
                self.status.setText(tr("{count} fields; {errors} unreadable previews.",
                                       count=len(page), errors=len(errors)))
            self._montage_jobs.submit(
                lambda: _overview_montage(page, channels, limits, cancelled,
                    preview_edge=min(240, resolution), labels=labels, **options), show_montage)

    def _plate_context(self):
        """What the plate thumbnails depend on: the plate, its filters and the display.

        Unlike :meth:`_visual_context`, opening a well or an image does not
        change it, so thumbnails decoded meanwhile are still installed.
        """
        try:
            channels, limits = self._choices()
        except (ValueError, TypeError):
            return None
        return (self._shown_plates(), self.rows.currentData(), self.columns.currentData(),
                self._display_generation, self.fov_axis.currentData(), self.time_axis.currentData(),
                self.z_axis.currentData(), self.z_mode.currentData(), channels, limits,
                self.preview_resolution.currentData(), self._enhancement, self.representatives.value())

    def _update_plate_thumbnails(self, channels, limits, options, resolution, cancelled):
        """Install cached plate thumbnails and decode the missing ones in parallel.

        Up to :data:`_OVERVIEW_THUMBNAIL_WORKERS` bounded batches run at once;
        the first batch is a single well, so the first image paints at once.
        Each shown plate is scanned only over the wells its own view shows.
        """
        if self._plate_panel.is_collapsed() or self._thumbnail_installs:
            return
        if not self._thumbnail_jobs.active_jobs():
            self._thumbnail_inflight.clear()
            self._thumbnail_first_inflight.clear()
        from PySide6.QtGui import QIcon

        spans = {}
        for plate, table in self._plate_tables.items():
            viewport = table.viewport()
            first_row = max(0, table.rowAt(0))
            last_row = table.rowAt(viewport.height() - 1)
            if last_row < 0:
                last_row = table.rowCount() - 1
            first_column = max(0, table.columnAt(0))
            last_column = table.columnAt(viewport.width() - 1)
            if last_column < 0:
                last_column = table.columnCount() - 1
            spans[plate] = (table, first_row, last_row, first_column, last_column)
        for address in tuple(self._installed_thumbnails):
            span = spans.get(address[0])
            position = self._well_positions.get(address)
            item = span[0].item(*position) if span is not None and position is not None else None
            if (item is None or not span[1] <= position[0] <= span[2]
                    or not span[3] <= position[1] <= span[4]):
                if item is not None:
                    item.setIcon(QIcon())
                    item.setData(Qt.UserRole + 1, None)
                self._installed_thumbnails.discard(address)
        free = _OVERVIEW_THUMBNAIL_WORKERS - self._thumbnail_jobs.active_jobs()
        if free <= 0:
            return
        wanted = free * _OVERVIEW_THUMBNAIL_BATCH
        inflight = self._thumbnail_inflight
        queued = {entry[1] for entry in self._thumbnail_installs}
        shown = self.representatives.value()
        display_key = self._display_options_key(options)
        requests, later = [], []
        for plate, (table, first_row, last_row, first_column, last_column) in spans.items():
            ratio = device_ratio(table.viewport())
            extent = self._plate_extents.get(table, self._plate_cell_extent)
            edge = (None if self._quality_original()
                    else min(resolution, max(64, math.ceil(max(extent or (160, 160)) * ratio))))
            folded_rows, folded_columns = self._collapsed_lines(plate)
            for row in range(first_row, last_row + 1):
                for column in range(first_column, last_column + 1):
                    item = table.item(row, column)
                    if item is None or row in folded_rows or column in folded_columns:
                        continue
                    address = (plate, item.data(Qt.UserRole))
                    field = self._well_representatives.get(address)
                    if field is None:
                        continue
                    representatives = tuple(self._well_fields[address][:shown])
                    key = (tuple(record.identity for record in representatives), channels, limits, ratio, edge,
                           display_key)
                    first = ((field.identity,),) + key[1:]
                    installed = item.data(Qt.UserRole + 1)
                    if installed == key or key in queued:
                        continue
                    if key in self._thumbnail_cache:
                        self._set_thumbnail(address, key, self._thumbnail_cache[key])
                        continue
                    if installed != first and first in self._thumbnail_cache:
                        self._set_thumbnail(address, first, self._thumbnail_cache[first])
                        installed = first
                    if installed == first:
                        if key not in inflight:
                            later.append((address, key, field, representatives, edge))
                        continue
                    if first not in inflight and len(requests) < wanted:
                        requests.append((address, first, field, (field,), edge))
        if not requests and not self._thumbnail_first_inflight:
            requests = later[:wanted]
        if not requests:
            return
        ratio = device_ratio(self.wells.viewport())
        context = self._plate_context()
        batch = _OVERVIEW_THUMBNAIL_BATCH
        if not self._installed_thumbnails:
            if self._thumbnail_jobs.active_jobs():
                return
            requests, free, batch = requests[:1], 1, 1
        while requests and free > 0:
            chunk, requests = requests[:batch], requests[batch:]
            batch, free = _OVERVIEW_THUMBNAIL_BATCH, free - 1
            inflight.update(key for _well, key, *_rest in chunk)
            self._thumbnail_first_inflight.update(
                key for _well, key, _field, representatives, _edge in chunk if len(representatives) == 1)
            self._submit_plate_thumbnails(chunk, channels, limits, options, resolution, cancelled, ratio, context)

    def _submit_plate_thumbnails(self, requests, channels, limits, options, resolution, cancelled, ratio, context):
        """Decode one bounded batch of plate thumbnails on a worker thread."""
        def read_thumbnails():
            """Read one batch of detached previews."""
            results = []
            for well, key, field, representatives, edge in requests:
                if cancelled.is_set():
                    break
                try:
                    if len(representatives) > 1:
                        pixels, errors = _overview_montage(representatives, channels, limits,
                            cancelled, preview_edge=min(240, resolution), **options)
                        error = "; ".join(message for _, message in errors)
                    else:
                        pixels = _overview_pixels(field, channels, limits, edge=edge, **options)
                        error = ""
                except Exception as exc:
                    pixels, error = None, str(exc)
                results.append((well, key, field, pixels, error))
            return results
        def done(results):
            """Install each current thumbnail and preserve individual read failures."""
            for _well, key, *_rest in requests:
                self._thumbnail_inflight.discard(key)
                self._thumbnail_first_inflight.discard(key)
            if cancelled.is_set():
                return
            for well, key, field, pixels, error in results:
                self._thumbnail_cache[key] = pixels
                while len(self._thumbnail_cache) > _OVERVIEW_THUMBNAIL_CACHE:
                    self._thumbnail_cache.popitem(last=False)
                self._thumbnail_installs.append((well, key, field, pixels, error, context, ratio))
            if self._thumbnail_installs:
                self._install_timer.start(0)
        self._thumbnail_jobs.submit(read_thumbnails, done)

    def _install_pending_thumbnails(self):
        """Install decoded plate thumbnails within a per-turn time budget.

        Eight workers finishing together delivered 64 thumbnails in one event
        turn; converting and attaching them all held the GUI past 50 ms. The
        finished batches queue them, and each timer turn installs for at most
        :data:`_OVERVIEW_INSTALL_BUDGET_S`, leaving the rest for the next. The
        plate is scanned for more work only once the queue is empty. A thumbnail is installed only while
        the plate context and device ratio it was decoded for still hold.
        """
        deadline = time.monotonic() + _OVERVIEW_INSTALL_BUDGET_S
        pending = self._thumbnail_installs
        while pending and time.monotonic() < deadline:
            well, key, field, pixels, error, context, ratio = pending.pop(0)
            if (self._cancelled.is_set() or self._plate_context() != context
                    or device_ratio(self.wells.viewport()) != ratio):
                continue
            self._set_thumbnail(well, key, pixels)
            if error:
                self.status.setText(tr("Could not read {field}: {error}",
                                       field=field.caption, error=error))
        if pending:
            self._install_timer.start(0)
        else:
            self._request_visuals()

    def _set_thumbnail(self, address, key, pixels):
        """Attach an image to one shown plate's well cell, only on the GUI thread.

        :param address: the well's ``(plate, well)``.
        """
        from PySide6.QtGui import QIcon
        position = self._well_positions.get(address)
        table = self._plate_tables.get(address[0])
        if position is None or table is None:
            return
        item = table.item(*position)
        if item is None or not table.viewport().rect().intersects(table.visualItemRect(item)):
            return
        item.setData(Qt.UserRole + 1, key)
        if pixels is not None:
            image = QImage(pixels.data, pixels.shape[1], pixels.shape[0],
                           pixels.strides[0], QImage.Format_RGB888).copy()
            pixmap = scaled_for(QPixmap.fromImage(image), table.viewport(), table.iconSize())
            item.setIcon(QIcon(pixmap))
            self._installed_thumbnails.add(address)
        else:
            item.setIcon(QIcon())
            self._installed_thumbnails.discard(address)

    def _refresh_image_if_pending(self, *_):
        """Coalesce rapid selections behind one active image decoder.

        :param _: unused image-job completion arguments.
        """
        if self._image_pending:
            self._image_pending = False
            item = self.fields.currentItem()
            if item is not None and self._individual:
                self._field(item, preserve_view=True)

    def _show_pixels(self, pixels, preserve_view=False):
        """Deep-copy RGB pixels and create pixmaps only on the GUI thread.

        :param pixels: detached RGB pixels from a worker or bounded cache.
        :param preserve_view: retain normalized center and user zoom on an upgrade.
        """
        from PySide6.QtCore import QPointF, QRectF
        previous = self.view._item
        keep = preserve_view and previous is not None and self.view._user_zoomed
        if keep:
            old_size = previous.pixmap().size()
            anchor = self.view.viewport().rect().center()
            center = self.view.mapToScene(anchor)
            scene_rect = self.view.sceneRect()
            normalized = QPointF(center.x() / old_size.width(), center.y() / old_size.height())
            transform = self.view.transform()
        image = QImage(pixels.data, pixels.shape[1], pixels.shape[0],
                       pixels.strides[0], QImage.Format_RGB888).copy()
        self._native_item = self._native_key = None
        self.view.set_pixmap(QPixmap.fromImage(image))
        self._update_image_metadata()
        if keep:
            x_ratio = image.width() / old_size.width()
            y_ratio = image.height() / old_size.height()
            self.view.setSceneRect(QRectF(scene_rect.x() * x_ratio, scene_rect.y() * y_ratio,
                                         scene_rect.width() * x_ratio, scene_rect.height() * y_ratio))
            self.view.setTransform(transform)
            self.view.scale(1. / x_ratio, 1. / y_ratio)
            target = QPointF(normalized.x() * image.width(), normalized.y() * image.height())
            self.view.centerOn(target)
            residual = self.view.viewportTransform().map(target) - QPointF(anchor)
            horizontal, vertical = self.view.horizontalScrollBar(), self.view.verticalScrollBar()
            horizontal.setValue(horizontal.value() + round(residual.x()))
            vertical.setValue(vertical.value() + round(residual.y()))
            self.view._user_zoomed = True
            self._request_native_detail()

    def _export_selection(self):
        """Export all explicit selections; sample only when selection mode is off."""
        scope = self.export_scope.currentData()
        if scope == "field":
            item = self.fields.currentItem()
            return (item.data(Qt.UserRole),) if item is not None and self._individual else ()
        if scope in ("plate", "well"):
            if scope == "well" and self._scope[1] is None:
                return ()
            time, z = self.time_axis.currentData(), self.z_axis.currentData()
            return tuple(field for field in self._fields
                if field.plate == self.plates.currentData()
                and (scope != "well" or field.well == self._scope[1])
                and (not getattr(field, "time_id", "") or field.time_id == time)
                and (not getattr(field, "z_id", "") or field.z_id == z))
        if scope == "selection" or self.use_selection.isChecked():
            fields = tuple(self._selected_fields.values())
            if fields:
                return fields
            time, z = self.time_axis.currentData(), self.z_axis.currentData()
            fields = tuple(field for field in self._fields
                if (field.plate, field.well) in self._selected_wells
                and (not field.time_id or field.time_id == self._selected_well_axes.get(
                    (field.plate, field.well), (time, z))[0])
                and (not field.z_id or field.z_id == self._selected_well_axes.get(
                    (field.plate, field.well), (time, z))[1]))
            if self.sample_selected.isChecked():
                return self._sample_fields(fields)
            return fields
        return self._sample_fields(self._visible)

    def _export(self):
        """Export explicit field/well selections or reproducible examples of this view."""
        from PySide6.QtWidgets import QFileDialog
        import threading
        if not self._geometry_valid:
            self.status.setText(str(self._geometry_error))
            return
        if not self._normalization_ready:
            return
        try:
            channels, limits = self._choices()
        except (ValueError, TypeError) as exc:
            self.status.setText(str(exc))
            return
        selected = self._export_selection()
        if not selected:
            self.status.setText(tr("Select fields or load a plate before exporting."))
            return
        if self._export_jobs.is_busy():
            self.status.setText(tr("Wait for the current export to finish or cancel it."))
            return
        if not self._metadata_source:
            self.status.setText(tr("Choose an image folder."))
            return
        from datetime import datetime
        folder = str(Path(self._metadata_source) / "results" /
                     ("overview_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
        self._cancelled.set()
        self._cancelled = threading.Event()
        if self._normalization_context is not None:
            self._display_generation += 1
            self._image_pending = False
            self._display_pending = False
            self._normalization_context = self._normalization_context._new_generation(self._cancelled)
            self._normalization_plan = self._normalization_context.provenance()
        cancelled, seed = self._cancelled, self.seed.value()
        measurements = self.measurements.isChecked()
        options = self._display_kwargs()
        options.update(load_limits=self._load_limits)
        options.update(quality_edge=self.pdf_quality.currentData(), dpi=self.pdf_dpi.currentData(),
                       conditions=dict(self._conditions), annotations=self._annotation_snapshot(),
                       field_options=self._selection_options(selected))
        self._export_jobs.submit(lambda: _overview_export(
            selected, folder, channels, limits, seed, cancelled, measurements, **options),
            lambda result: self._exported(result, measurements, folder, cancelled))

    def _canvas_snapshot(self):
        """What the views show now, frozen for rendering canvas images off the GUI thread."""
        try:
            channels, limits = self._choices()
        except _OverviewValidationError:
            channels, limits = (0,), None
        display = self._display_kwargs()
        shown = self._display_kwargs(view=True)
        for key in ("channel_limits", "image_overrides", "percentile_channels"):
            if key in shown:
                display[key] = shown[key]
        return _OverviewCanvasSnapshot(self._fields, channels, limits, display)

    def _canvas_well_fields(self, plate, well):
        """The fields of one well at the shown time and Z, in field order."""
        time, z = self.time_axis.currentData(), self.z_axis.currentData()
        fields = [field for field in self._fields if field.plate == plate and field.well == well
                  and (not field.time_id or field.time_id == time) and (not field.z_id or field.z_id == z)]
        return sorted(fields, key=lambda field: self._natural(field.field_id or field.field))

    def _canvas_channel_names(self):
        """The shown channels by the names on their buttons."""
        try:
            channels = self._choices()[0]
        except _OverviewValidationError:
            return []
        buttons = getattr(self, "_channel_buttons", {}) or {}
        return [buttons[channel].text() if channel in buttons else str(channel) for channel in channels]

    def _canvas_display_note(self):
        """Lines describing the display settings, for the last page of a guided export."""
        try:
            channels, limits = self._choices()
        except _OverviewValidationError:
            return ()
        plan = (self._normalization_context.provenance() if self._normalization_context is not None
                else self._normalization_plan)
        try:
            lines = list(_overview_pdf_header(channels, limits, self.seed.value(),
                                              None if self.advanced.isChecked() else plan, self._color_mode))
        except (KeyError, TypeError, ValueError):
            lines = list(_overview_pdf_header(channels, limits, self.seed.value(), None, self._color_mode))
        pixel = self._overview_options.get("pixel_size_um")
        if pixel:
            lines.append(tr("Pixel size: {size} µm", size=f"{pixel:g}"))
        return tuple([tr("Display settings")] + lines)

    def _export_menu(self, anchor=None):
        """Offer the exports: guided, custom canvas, contact sheet; returns the menu.

        :param anchor: the button the menu opens under; the pointer otherwise.
        """
        from PySide6.QtGui import QCursor
        from PySide6.QtWidgets import QMenu

        menu = QMenu(self)
        menu.setObjectName("OverviewExportMenu")
        menu.setToolTipsVisible(True)
        for caption, tooltip, slot in (
                ("Guided export…", "Choose wells and fields, place each well on the page and choose what is "
                 "printed around it, then preview and write a PDF.", self._open_guided_export),
                ("Custom export…", "Draw text boxes and drop images on a page grid, then write a PDF.",
                 self._open_custom_export),
                ("Contact sheet PDFs", "Twelve fields per page from the export scope, with the display "
                 "settings printed above them.", self._export)):
            action = menu.addAction(tr(caption))
            action.setToolTip(tr(tooltip))
            action.triggered.connect(functools.partial(self._run_export_action, slot))
        if anchor is not None and anchor.isVisible():
            menu.popup(anchor.mapToGlobal(anchor.rect().bottomLeft()))
        else:
            menu.popup(QCursor.pos())
        self._export_menu_shown = menu
        return menu

    def _run_export_action(self, slot, *_):
        """Run an export chosen from the Export menu."""
        slot()

    def _open_guided_export(self):
        """Open the guided export in the export section at the far right; returns it."""
        if not self._fields:
            self.status.setText(tr("Select fields or load a plate before exporting."))
            return None
        dialog = _OverviewGuidedExport(self)
        self._guided_export = dialog
        self._show_export_dock(tr("Guided export"), dialog)
        return dialog

    def _open_custom_export(self, layout=None):
        """Open the Custom export canvas in the export section, empty or with ``layout``; returns it.

        While it is open, Overview images come to it in every way
        :class:`_OverviewCanvasDrags` lists.
        """
        from .picture_export import _CanvasComposer

        composer = _CanvasComposer(layout, _OverviewCanvasSource(self), self._metadata_source or "", self,
                                   actions=(("Add selected", "Place the selected fields, or the fields of the "
                                             "selected wells, on this page.", self._canvas_add_selection),))
        drags = getattr(self, "_canvas_drags", None)
        if drags is None:
            drags = self._canvas_drags = _OverviewCanvasDrags(self)
        composer.finished.connect(drags.unwatch)
        self._canvas_composer = composer
        self._show_export_dock(tr("Custom export"), composer, tr(
            "Drag images from the plate, well or image views onto the page; press and hold an image, or use the "
            "Drag to canvas handle, when a plain drag would pan. Hold {keys} and click any image to add it. Image "
            "files can be dropped here from the file manager.", keys=_overview_pick_keys()))
        drags.watch()
        composer.status.setText(tr("Hold {keys} and click any image in the Overview to add it to this page.",
                                   keys=_overview_pick_keys()))
        return composer

    def _canvas_pick(self, images):
        """Add images clicked in pick mode to the open Custom export page; returns the boxes."""
        composer = getattr(self, "_canvas_composer", None)
        if composer is None or self._export_dock is None or self._export_dock.content is not composer:
            return []
        boxes = composer.add_images(images)
        composer.status.setText(tr("Added {count} images to page {page}.", count=len(boxes),
                                   page=composer.page_index + 1))
        return boxes

    def _export_settings_body(self):
        """The Overview tab's settings splitter, or ``None`` outside the tab."""
        from .collapsible_splitter import CollapsibleSplitter

        widget = self.parentWidget()
        while widget is not None and not isinstance(widget, _OverviewModulePage):
            widget = widget.parentWidget()
        body = getattr(getattr(widget, "_screen", None), "_body_splitter", None)
        return body if isinstance(body, CollapsibleSplitter) and body.pane("Settings") is not None else None

    def _show_export_dock(self, title, content, hint=""):
        """Show ``content`` as the export section at the far right and fold the settings away.

        Any export already shown is closed first; the settings' state from
        before the first export is what Close restores.

        :returns: the :class:`_OverviewExportDock`.
        """
        self._close_export_dock(restore=False)
        dock = _OverviewExportDock(title, content, hint)
        dock.closed.connect(self._close_export_dock)
        content.finished.connect(functools.partial(self._export_content_closed, content))
        self._export_dock = dock
        body = self._export_settings_body()
        if body is not None and self._export_settings_state is None:
            self._export_settings_state = (body.is_collapsed("Settings"), body.sizes())
            body.set_collapsed("Settings", True)
        from ..theme import ensure_widget_qss_applied, register_widget_qss

        register_widget_qss("OverviewExportDock", _overview_export_qss, replace=True)
        ensure_widget_qss_applied("OverviewExportDock", root=self)
        host = self._export_host()
        host.addWidget(dock)
        host.setCollapsible(host.indexOf(dock), False)
        self._size_export_dock()
        QTimer.singleShot(0, self._size_export_dock)
        return dock

    def _export_host(self):
        """The splitter that holds the plate, well and image views on the left and the export section on the right.

        It is put in the views' place the first time an export opens, so the
        views keep their own splitter and sizes.
        """
        from PySide6.QtWidgets import QSplitter

        host = getattr(self, "_export_splitter", None)
        if host is None:
            host = QSplitter(Qt.Horizontal)
            host.setObjectName("OverviewExportSplitter")
            host.setChildrenCollapsible(False)
            self.layout().replaceWidget(self._splitter, host)
            host.addWidget(self._splitter)
            host.show()
            self._export_splitter = host
        return host

    def _size_export_dock(self):
        """Give the export section three fifths of the width, and at least what its controls need."""
        from shiboken6 import isValid

        dock = self._export_dock
        host = getattr(self, "_export_splitter", None)
        if dock is None or host is None or not isValid(dock) or host.indexOf(dock) < 0:
            return
        total = max(sum(host.sizes()), host.width(), 1)
        wide = min(max(total * 3 // 5, _OVERVIEW_EXPORT_DOCK_MIN_WIDTH), total * 4 // 5)
        host.setSizes([total - wide, wide])

    def _export_content_closed(self, content, *_):
        """Take the export section away when the export in it closes itself."""
        dock = self._export_dock
        if dock is not None and dock.content is content:
            self._close_export_dock()

    def _close_export_dock(self, restore=True):
        """Close the export section; with ``restore`` the settings come back as they were."""
        from shiboken6 import isValid

        dock, self._export_dock = self._export_dock, None
        if dock is not None and isValid(dock):
            content = dock.content
            drags = getattr(self, "_canvas_drags", None)
            if drags is not None and content is getattr(self, "_canvas_composer", None):
                drags.unwatch()
            if isValid(content):
                content.close()
            dock.hide()
            dock.setParent(None)
            dock.deleteLater()
        if not restore or self._export_settings_state is None:
            return
        collapsed, sizes = self._export_settings_state
        self._export_settings_state = None
        body = self._export_settings_body()
        if body is None:
            return
        body.set_collapsed("Settings", collapsed, by_user=True)
        if not collapsed and len(sizes) == len(body.sizes()):
            body.setSizes(sizes)

    def _canvas_add_selection(self, composer):
        """Place the selected fields, or the selected wells' fields, on the canvas page."""
        fields = list(self._selected_fields.values())
        if not fields:
            for plate, well in sorted(self._selected_wells):
                fields.extend(self._canvas_well_fields(plate, well)[:_OVERVIEW_GUIDED_MAX_WELL_FIELDS])
        if not fields:
            composer.status.setText(tr("Select fields or wells in the Overview first."))
            return []
        pixel = self._overview_options.get("pixel_size_um")
        return composer.add_images([_overview_canvas_image(field, pixel) for field in fields])

    def _exported(self, result, measurements=False, folder="", cancelled=None):
        """Report output and unreadable fields without hiding either."""
        paths, errors = result
        message = tr("Wrote {count} PDF pages; {errors} unreadable fields.",
                     count=len(paths), errors=len(errors))
        if measurements:
            if cancelled is not None and cancelled.is_set():
                message += " " + tr("Measurements CSV was not published because export was cancelled.")
            else:
                message += " " + tr("Raw whole-image measurements: {path}",
                    path=str(Path(folder) / "overview_measurements.csv"))
        self.status.setText(message)

    def closeEvent(self, event):
        """Cancel generations when closing; workers never touch the closed widgets.

        :param event: the native close event passed to the base dialog.
        """
        self._retire()
        super().closeEvent(event)

    def _loading(self):
        """Whether this shown browser is still indexing, preparing or decoding images."""
        if getattr(self, "_module_actions_retired", False) or not self.isVisible():
            return False
        if not self._normalization_ready:
            return True
        return any(runner.is_busy() for runner in (
            self._index_jobs, self._image_jobs, self._thumbnail_jobs,
            self._field_thumbnail_jobs, self._montage_jobs))

    def _mount_navigation_bar(self):
        """Put plate and field navigation above the plate, as labelled groups of buttons.

        Navigation returns to the plate; Wells fits the plate or makes the
        wells smaller (minus) or larger (plus); Fields pages through a well's
        fields and compares the selected ones. Opening a well or an image is
        a double click, so those two buttons are not shown.

        :returns: the bar.
        """
        from .flow import FlowHost, FlowLayout

        fit, larger, smaller = self._plate_zoom_buttons
        for button, text, tooltip in ((smaller, "−", "Smaller wells"), (larger, "+", "Larger wells")):
            button.setText(text)
            button.setToolTip(tr(tooltip))
            button.setAccessibleName(tr(tooltip))
        fit.setToolTip(tr("Fit the whole plate in view"))
        groups = (("Navigation", (self._plate_back_button,)),
                  ("Plates", (*self._plate_buttons, *self._plates_view_controls)),
                  ("Wells", (fit, smaller, larger)),
                  ("Fields", (*self._paging_buttons, self._compare_fields_button)))
        bar = FlowHost(self)
        bar.setObjectName("FieldOverviewNavigation")
        flow = FlowLayout(bar)
        for caption, buttons in groups:
            group = QWidget(bar)
            row = QHBoxLayout(group)
            row.setContentsMargins(0, 0, 8, 0)
            row.setSpacing(4)
            title = QLabel(tr(caption))
            title.setObjectName("FieldOverviewNavigationGroup")
            font = title.font()
            font.setBold(True)
            title.setFont(font)
            row.addWidget(title)
            for button in buttons:
                button.show()
                row.addWidget(button)
            flow.addWidget(group)
        for button in (self._open_well_button, self._open_images_button):
            button.hide()
        holder = QWidget(self)
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        index = self._splitter.indexOf(self._levels)
        layout.addWidget(bar)
        layout.addWidget(self._levels, 1)
        self._splitter.insertWidget(max(0, index), holder)
        self._navigation_bar = bar
        return bar

    def _mounted_control_lost(self, *_):
        """Stop for good once a control mounted in a setting row is deleted.

        The plate, axis and navigation controls live in the Overview tab's
        setting rows. Deleted with those rows while this browser still ran,
        a display refresh read a deleted combo box (GitHub #165); the browser
        now retires and ignores later refreshes instead.
        """
        self._controls_lost = True
        self._retire()

    def _retire(self):
        """Stop owned image generations even when a page is already hidden."""
        from ..gil_priority import release_background_imports

        release_background_imports(self)
        self._module_actions_retired = True
        self._module_action_fit_timer.stop()
        self._options_window.close()
        self._close_export_dock()
        drags = getattr(self, "_canvas_drags", None)
        if drags is not None:
            drags.unwatch()
        self._cancel()
        self._visual_timer.stop()
        self._prepare_timer.stop()
        self._install_timer.stop()
        self._thumbnail_installs.clear()
        self._native_timer.stop()
        self._measurement_cancelled.set()
        for runner in (self._index_jobs, self._image_jobs, self._export_jobs,
                       self._thumbnail_jobs, self._montage_jobs, self._field_thumbnail_jobs,
                       self._metadata_jobs, self._measurement_jobs, self._native_jobs,
                       self._image_stack_jobs, self._group_jobs, self._readout_jobs, self._levels_jobs):
            runner.shutdown(timeout_ms=0)
        if self._image_settings is not None:
            from shiboken6 import isValid

            if isValid(self._image_settings):
                self._image_settings.close()
            self._image_settings = None
        self._clear_pixel_readout()


def _overview_workspace_qss(palette, opacity):
    """Let embedded overview content inherit the owning module's backdrop.

    The metadata step's filename regex keeps a full frame: a long pattern
    runs to the field's right end, where the field fade would leave it with
    no visible edge.
    """
    return ("QWidget#FieldOverviewWorkspace, QWidget#FieldOverviewBody, "
            "QWidget#FieldOverviewMetadata { background: transparent; }\n"
            + _overview_regex_qss(palette))


def _overview_regex_qss(palette):
    """The framed look of the metadata step's filename regex field.

    :param palette: the active palette.
    :returns: QSS for ``QLineEdit#OverviewWizardRegex``.
    """
    from ..theme import RADIUS

    return (f"QLineEdit#OverviewWizardRegex {{ background-color: {palette['surface_alt']}; "
            f"border: 1px solid {palette['border']}; border-radius: {RADIUS['sm']}px; }}\n"
            f"QLineEdit#OverviewWizardRegex:focus {{ border: 1px solid {palette['accent']}; }}\n"
            f"QLineEdit#OverviewWizardRegex:disabled {{ background-color: {palette['surface']}; "
            f"border-color: {palette['border_soft']}; }}")


class _OverviewWorkspace(QWidget):
    """Retain overview stages inside the owning module's existing panels."""

    def __init__(self, owner, source):
        """Reuse the owning module's panels without opening another top-level window."""
        from PySide6.QtWidgets import QStackedWidget
        from ..theme import register_widget_qss, ensure_widget_qss_applied
        super().__init__(owner)
        self._owner, self.source = owner, source
        self._previous_workspace = None
        self._module_mounted = False
        self._active_child = None
        self._inactive_page_policies = {}
        self._saved_settings = None
        self._module_hidden = []
        self._module_panes = []
        self._module_figures = None
        self._active_actions = None
        self.setObjectName("FieldOverviewWorkspace")
        self.setAutoFillBackground(False)
        layout = QVBoxLayout(self)
        navigation = QHBoxLayout()
        self.back_button = QPushButton(tr("Back"))
        self.back_button.clicked.connect(self._back)
        self.back_button.hide()
        self._back_fit_busy = False
        self._back_fit_timer = QTimer(self)
        self._back_fit_timer.setSingleShot(True)
        self._back_fit_timer.timeout.connect(self._fit_back_button)
        self.back_button.installEventFilter(self)
        self._back_fit_timer.start(0)
        navigation.addWidget(self.back_button)
        navigation.addWidget(QLabel(tr("Field-of-view overview")), 1)
        layout.addLayout(navigation)
        self.stack = QStackedWidget(self)
        self.stack.currentChanged.connect(self._activate_child)
        layout.addWidget(self.stack, 1)
        register_widget_qss("FieldOverviewWorkspace", _overview_workspace_qss, replace=True)
        ensure_widget_qss_applied("FieldOverviewWorkspace", root=self)

    def _fit_back_button(self):
        """Refresh the owned Back action's native text floor without fixing its size."""
        self._back_fit_busy = True
        try:
            _fit_overview_action_buttons((self.back_button,))
        finally:
            self._back_fit_busy = False

    def eventFilter(self, watched, event):
        """Coalesce the owned Back action's font/style changes after Qt applies them."""
        if (watched is self.back_button and not self._back_fit_busy
                and event.type() in (QEvent.FontChange, QEvent.StyleChange, QEvent.Polish)):
            self._back_fit_timer.start(0)
        return super().eventFilter(watched, event)

    def show_page(self):
        """Reuse module panels, falling back to the shared page host for bare callers."""
        from ..screens.map_barcodes import show_as_page
        if self._mount_module():
            self._owner.show()
            self.show()
            return
        if self._owner.layout() is None:
            layout = QVBoxLayout(self._owner)
            body = QWidget(self._owner)
            layout.addWidget(body, 1)
            if self._owner.isVisible():
                body.show()
        show_as_page(self, self._owner, tr("Field-of-view overview"))
        self._owner.show()
        self.show()

    def _mount_module(self):
        """Mount the real module settings and image area while retaining its chrome."""
        from PySide6.QtWidgets import QStackedWidget
        owner = self._owner
        settings = getattr(owner, "_settings_scroll", None)
        figures = getattr(owner, "_figures_card", None)
        images = getattr(owner, "_body_stack", None)
        if settings is None or (figures is None and images is None):
            return False
        if self._module_mounted:
            self._module_stack.setCurrentWidget(self)
            return True
        if figures is not None:
            target = getattr(owner, "_overview_content_stack", None)
            if target is None:
                target = QStackedWidget(figures.body)
                original = QWidget()
                original_layout = QVBoxLayout(original)
                original_layout.setContentsMargins(0, 0, 0, 0)
                while figures.body_layout.count():
                    item = figures.body_layout.takeAt(0)
                    if item.widget() is not None:
                        original_layout.addWidget(item.widget())
                    else:
                        original_layout.addItem(item)
                target.addWidget(original)
                figures.body_layout.addWidget(target, 1)
                owner._overview_content_stack = target
        else:
            target = images
        self._module_stack = target
        pages = getattr(owner, "_fold_pages", None)
        if pages is not None:
            pages.setCurrentIndex(0)
        self._previous_module_image = target.currentWidget()
        self._previous_module_image_policy = (self._previous_module_image.sizePolicy()
                                               if self._previous_module_image is not None else None)
        if self._previous_module_image is not None:
            self._previous_module_image.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._saved_settings = settings.takeWidget()
        self._saved_settings.setParent(owner)
        self._saved_settings.hide()
        self._module_mounted = True
        for name in ("_actions_body", "_settings_search", "_tool_row", "_bottom_bar", "_image_footer"):
            widget = getattr(owner, name, None)
            if widget is not None:
                self._module_hidden.append((widget, widget.isHidden()))
                widget.hide()
        target.addWidget(self)
        target.setCurrentWidget(self)
        for name, panels in (("_body_splitter", ("Settings",)),
                ("_runtime_splitter", ("Figures", "Console", "System", "Actions"))):
            splitter = getattr(owner, name, None)
            if splitter is not None:
                for pane in panels:
                    if splitter.pane(pane) is not None:
                        self._module_panes.append((splitter, pane, splitter.is_collapsed(pane)))
        if figures is not None:
            pane = owner._runtime_splitter.pane("Figures")
            self._module_figures = figures, figures.isHidden(), figures.minimumHeight(), pane.minimum, pane.extent
            figures.setMinimumHeight(0)
            pane.minimum = 0
            figures.show()
        for splitter, pane, _collapsed in self._module_panes:
            if pane in ("Settings", "Figures"):
                splitter.set_collapsed(pane, False)
        self._activate_child()
        return True

    def _return_child_settings(self):
        """Return borrowed controls to their original retained widget."""
        child = self._active_child
        if child is None:
            return
        if isinstance(child, _FieldOverview):
            child._options_window.hide()
            child._options_closed()
        settings = self._owner._settings_scroll
        content = settings.takeWidget()
        if content is not None:
            scroll = child._controls_scroll if isinstance(child, _FieldOverview) else child._settings_scroll
            scroll.setWidget(content)
        if isinstance(child, _FieldOverview):
            child._settings_host = None
        if self._active_actions is not None:
            self._active_actions.setParent(child)
            self._active_actions.hide()
            self._active_actions = None
        self._active_child = None

    def _activate_child(self, *_):
        """Show the active stage's categories and safe actions in the normal panels."""
        from ..theme import clear_container_surfaces, make_transparent
        current = self.stack.currentWidget()
        self.back_button.setVisible(isinstance(current, _FieldOverview))
        if not self._module_mounted or current is self._active_child:
            return
        from PySide6.QtWidgets import QSizePolicy
        if self._active_child is not None:
            previous = self._active_child
            self._inactive_page_policies[previous] = previous.sizePolicy()
            previous.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        if current in self._inactive_page_policies:
            current.setSizePolicy(self._inactive_page_policies.pop(current))
        self._return_child_settings()
        scroll = current._controls_scroll if isinstance(current, _FieldOverview) else current._settings_scroll
        content = scroll.takeWidget()
        clear_container_surfaces(current)
        clear_container_surfaces(content)
        clear_container_surfaces(self)
        make_transparent(self, self.stack, self._module_stack, current, content, current._module_actions)
        if not content.property("spacrLabelsWrapped"):
            for label in content.findChildren(QLabel):
                label.setWordWrap(True)
            content.setProperty("spacrLabelsWrapped", True)
        self._owner._settings_scroll.setWidget(content)
        if isinstance(current, _FieldOverview):
            current._settings_host = self._owner._settings_scroll
        else:
            current._settings_scroll.hide()
        section = getattr(self._owner, "_actions_section", None)
        if section is not None:
            actions = current._module_actions if isinstance(current, _FieldOverview) else current._host_actions
            section.body_layout.addWidget(actions)
            actions.show()
            self._active_actions = actions
        elif isinstance(current, _FieldOverview):
            current.layout().addWidget(current._module_actions)
            current._module_actions.show()
            self._active_actions = current._module_actions
        self._active_child = current

    def _leave_page(self):
        """Restore the exact host settings, image and action widgets on return."""
        from ..screens.map_barcodes import hide_as_page
        if self._module_mounted:
            self._return_child_settings()
            self._owner._settings_scroll.setWidget(self._saved_settings)
            self._saved_settings = None
            if self._previous_module_image_policy is not None:
                self._previous_module_image.setSizePolicy(self._previous_module_image_policy)
            self._module_stack.setCurrentWidget(self._previous_module_image)
            self._module_stack.removeWidget(self)
            self._module_mounted = False
            for child, policy in self._inactive_page_policies.items():
                child.setSizePolicy(policy)
            self._inactive_page_policies.clear()
            for widget, hidden in self._module_hidden:
                widget.setVisible(not hidden)
            self._module_hidden.clear()
            if self._module_figures is not None:
                figures, hidden, minimum, pane_minimum, extent = self._module_figures
                figures.setMinimumHeight(minimum)
                pane = self._owner._runtime_splitter.pane("Figures")
                pane.minimum, pane.extent = pane_minimum, extent
                figures.setVisible(not hidden)
                self._module_figures = None
            for splitter, pane, collapsed in self._module_panes:
                splitter.set_collapsed(pane, collapsed)
            self._module_panes.clear()
        else:
            hide_as_page(self, self._owner)
        self.hide()
        if not self._owner.isAncestorOf(self):
            self.setParent(self._owner, Qt.Widget)

    def _back(self):
        """Return through field, well, plate and the owning module."""
        current = self.stack.currentWidget()
        if isinstance(current, _FieldOverview) and current._individual:
            current._individual = False
            current._comparison_signature = None
            current._filter()
        elif isinstance(current, _FieldOverview) and current._scope[1] is not None:
            current._back()
        else:
            self._leave_page()

    def hideEvent(self, event):
        """Hide the independent options window when its workspace leaves view."""
        current = self.stack.currentWidget()
        if isinstance(current, _FieldOverview):
            current._options_window.hide()
            current._options_closed()
        super().hideEvent(event)

    def _metadata_rejected(self):
        """Restore a validated source when replacement metadata is cancelled."""
        previous = self._previous_workspace
        if previous is None:
            self._back()
            return
        self._previous_workspace = None
        self._leave_page()
        self._owner._field_overview_workspace = previous
        self._owner._field_overview_wizard = previous.stack.widget(0)
        previous.show_page()
        self.deleteLater()

    def _retire(self):
        """Retire owned child jobs independently of native hidden-widget closes."""
        self._back_fit_timer.stop()
        previous = self._previous_workspace
        self._previous_workspace = None
        if previous is not None:
            previous._retire()
            previous.deleteLater()
        for index in range(self.stack.count()):
            child = self.stack.widget(index)
            if isinstance(child, _FieldOverview):
                child._retire()
            else:
                child.reject()

    def closeEvent(self, event):
        """Retire the retained child jobs when replacing or closing the workspace."""
        self._retire()
        self._leave_page()
        super().closeEvent(event)


def _open_field_overview(owner, settings=None):
    """Open metadata and the image browser as a retained embedded submodule."""
    settings = dict(settings or {})
    source = _overview_source_path(settings.get("source") or settings.get("src"))
    previous = getattr(owner, "_field_overview_workspace", None)
    if previous is not None and previous.source == source:
        current = previous.stack.currentWidget()
        if not isinstance(current, _OverviewMetadataWizard) or not current._finished:
            previous.show_page()
            return owner._field_overview_wizard
    retained = None
    if previous is not None:
        if isinstance(previous.stack.currentWidget(), _FieldOverview):
            retained = previous
        else:
            retained = previous._previous_workspace
            previous._previous_workspace = None
            previous._retire()
            previous.deleteLater()
        previous._leave_page()
    workspace = _OverviewWorkspace(owner, source)
    workspace._previous_workspace = retained
    owner._field_overview_workspace = workspace
    wizard = _OverviewMetadataWizard(source, settings, workspace)
    wizard._startup_timer.stop()
    wizard.setWindowFlags(Qt.Widget)
    wizard.setObjectName("FieldOverviewMetadata")
    wizard.setAutoFillBackground(False)
    workspace.stack.addWidget(wizard)
    owner._field_overview_wizard = wizard

    def _show_browser():
        """Replace metadata choices with original-image navigation on the same page."""
        config = wizard.configuration()
        wizard.rejected.disconnect(workspace._metadata_rejected)
        if workspace._previous_workspace is not None:
            old = workspace._previous_workspace
            workspace._previous_workspace = None
            old._retire()
            old.deleteLater()
        dialog = _FieldOverview(workspace, config)
        dialog.setWindowFlags(Qt.Widget)
        dialog.setObjectName("FieldOverviewBody")
        dialog.setAutoFillBackground(False)
        owner._field_overview = dialog
        workspace.stack.addWidget(dialog)
        workspace.stack.setCurrentWidget(dialog)
        dialog._indexed(tuple(wizard._fields))

    wizard.accepted.connect(_show_browser)
    wizard.rejected.connect(workspace._metadata_rejected)
    workspace.show_page()
    wizard._startup_timer.start(0)
    return wizard


_OVERVIEW_SETTING_CONTROLS = (
    ("src", "source"), ("custom_regex", "regex"),
    ("overview_channels", "channels"),
    ("overview_preview_resolution", "preview_resolution"),
    ("overview_z_mode", "z_mode"), ("overview_wheel", "wheel_axis"),
    ("overview_representatives", "representatives"),
    ("overview_normalize", "normalise"),
    ("overview_percentile_low", "low"), ("overview_percentile_high", "high"),
    ("overview_advanced_normalization", "advanced"),
    ("overview_normalization_method", "normalization_method"),
    ("overview_advanced_low", "advanced_low"),
    ("overview_advanced_high", "advanced_high"),
    ("overview_fixed_low", "manual_low"), ("overview_fixed_high", "manual_high"),
    ("overview_background", "background"), ("overview_gamma", "gamma"),
    ("overview_reference_channel", "reference_channel"),
    ("overview_reference_epsilon", "reference_epsilon"),
    ("overview_representative_method", "representative_method"),
    ("overview_export_scope", "export_scope"),
    ("overview_pdf_quality", "pdf_quality"), ("overview_pdf_dpi", "pdf_dpi"),
    ("overview_sample_grouping", "grouping"),
    ("overview_sample_count", "count"), ("overview_sample_seed", "seed"),
    ("overview_random_sample", "random"),
    ("overview_export_selection", "use_selection"),
    ("overview_sample_selected", "sample_selected"),
    ("overview_measurements_csv", "measurements"),
    ("overview_workers", "workers"),
)

_OVERVIEW_CHOSEN_BY_METADATA = frozenset({
    "src", "custom_regex", "overview_channels", "overview_z_mode",
    "overview_representatives", "overview_percentile_low",
    "overview_percentile_high"})

_OVERVIEW_IMAGE_SETTINGS = ("overview_tile_size", "overview_tile_spacing",
                            "overview_enhance_channels",
                            "overview_normalize_channels", "overview_color_mode",
                            "overview_custom_colors",
                            "overview_max_wells", "overview_max_fields", "overview_normalization_scope",
                            "overview_ink_intensity")

_OVERVIEW_STAGED_SETTINGS = frozenset({
    "overview_normalize", "overview_percentile_low", "overview_percentile_high",
    "overview_normalize_channels", "overview_normalization_scope",
    "overview_advanced_normalization", "overview_normalization_method",
    "overview_advanced_low", "overview_advanced_high", "overview_fixed_low", "overview_fixed_high",
    "overview_background", "overview_gamma", "overview_reference_channel", "overview_reference_epsilon",
    "overview_enhance_channels", "overview_color_mode", "overview_custom_colors", "overview_ink_intensity"})
"""Display and normalization settings that wait for Apply; the ``enhance_*`` settings wait too.

Every other Overview setting is applied as it changes: the source, the
shown channels, preview resolution, opening at Original, Z mode, wheel,
fields per well, load limits, the tile size and spacing, and the
comparison and export settings.
"""


def _overview_channel_targets(value):
    """Read an 'all' or comma-separated channel setting.

    :param value: the setting value.
    :returns: the channels as a tuple, or ``None`` for all channels.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return (value,)
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        text = str(value).strip()
        if not text or text.lower() == "all":
            return None
        items = [part.strip() for part in text.split(",") if part.strip()]
    targets = tuple(int(item) if str(item).lstrip("-").isdigit() else item for item in items)
    return targets or None


def _overview_channel_list_error(value, known=()):
    """Check a typed channel list, such as ``overview_normalize_channels``.

    :param value: ``'all'``, ``'none'``, or channel numbers separated by
        commas (``'1,2,3,4'``); an int or a list of ints is accepted too.
    :param known: the source's channel IDs; when given, every number must be one of them.
    :returns: ``''`` when the list is valid, otherwise the reason, ready to show.
    """
    if isinstance(value, bool):
        return tr("Type channel numbers separated by commas, such as 1,2,3,4, or all.")
    if isinstance(value, int):
        items = [str(value)]
    elif isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value]
    else:
        text = str(value or "").strip()
        if text.lower() in ("all", "none"):
            return ""
        items = [part.strip() for part in text.split(",")]
    if not items or any(not item for item in items):
        return tr("Type channel numbers separated by commas, such as 1,2,3,4, or all.")
    channels = [str(channel) for channel in known or ()]
    wrong = [item for item in items if not item.isdigit() and item not in channels]
    if wrong:
        return tr("{value} is not a channel number; type channel numbers separated by commas, "
                  "such as 1,2,3,4, or all.", value=wrong[0])
    missing = [item for item in items if channels and item not in channels]
    if missing:
        return tr("Channel {channel} is not in this source; its channels are {channels}.",
                  channel=missing[0], channels=", ".join(channels))
    return ""


def _overview_control_value(control):
    """Read a browser control as the plain value its setting holds.

    :param control: a line edit, check box, spin box or combo box.
    :returns: the value; a combo's ``None`` data reads as ``"full"``.
    """
    from PySide6.QtWidgets import QAbstractSpinBox, QLineEdit

    if isinstance(control, QLineEdit):
        return control.text()
    if isinstance(control, QCheckBox):
        return control.isChecked()
    if isinstance(control, QComboBox):
        value = control.currentData()
        return "full" if value is None else value
    if isinstance(control, QAbstractSpinBox):
        return control.value()
    return None


def _overview_set_control(control, value):
    """Write a setting's value into its browser control.

    :param control: a line edit, check box, spin box or combo box.
    :param value: the setting's value.
    """
    from PySide6.QtWidgets import QAbstractSpinBox, QLineEdit

    if isinstance(control, QLineEdit):
        text = "" if value is None else (",".join(str(v) for v in value)
                                         if isinstance(value, (list, tuple)) else str(value))
        if control.text() != text:
            control.setText(text)
            control.editingFinished.emit()
    elif isinstance(control, QCheckBox):
        control.setChecked(bool(value))
    elif isinstance(control, QComboBox):
        for index in range(control.count()):
            data = control.itemData(index)
            if data == value or str("full" if data is None else data) == str(value):
                control.setCurrentIndex(index)
                return
    elif isinstance(control, QAbstractSpinBox):
        try:
            control.setValue(type(control.value())(value))
        except (TypeError, ValueError):
            pass


def _overview_control_changed(control, handler):
    """Connect ``handler`` to the signal a browser control commits with.

    :param control: the browser control.
    :param handler: called without arguments after each change.
    """
    from PySide6.QtWidgets import QDoubleSpinBox, QLineEdit, QSpinBox

    if isinstance(control, QLineEdit):
        control.editingFinished.connect(handler)
    elif isinstance(control, QCheckBox):
        control.toggled.connect(lambda *_: handler())
    elif isinstance(control, QComboBox):
        control.currentIndexChanged.connect(lambda *_: handler())
    elif isinstance(control, (QSpinBox, QDoubleSpinBox)):
        control.valueChanged.connect(lambda *_: handler())


class _OverviewSettingsBridge:
    """Keep the Overview module's setting rows and its browser in step.

    The rows are the module's normal settings; the browser keeps its own
    controls hidden and follows the rows, and a value the browser changes
    itself (the metadata step, a restored annotation set) is written back.
    """

    def __init__(self, screen, browser):
        """Bind every Overview setting the browser reads.

        :param screen: the Overview module screen.
        :param browser: the plate browser it shows.
        """
        self._screen, self._browser = screen, browser
        self._model = screen._settings_model
        self._syncing = False
        self._pairs = []
        self._connections = []
        self._detached = False
        self._staged = set()
        self._applying = False
        self._writing_display = False
        self.apply_bar = None
        browser.destroyed.connect(self._detach)
        for key, attribute in _OVERVIEW_SETTING_CONTROLS:
            row = self._model._widgets.get(key)
            control = getattr(browser, attribute, None)
            if row is None or control is None:
                continue
            self._pairs.append((key, row, control))
            if key in _OVERVIEW_CHOSEN_BY_METADATA:
                self._to_setting(key, control)
            else:
                self._to_control(key, control)
            self._listen(row, functools.partial(self._row_changed, key, control))
            _overview_control_changed(control, functools.partial(self._to_setting, key, control))
        from spacr.settings import _enhancement_keys

        self._image_keys = (*_OVERVIEW_IMAGE_SETTINGS, *_enhancement_keys())
        self._staged_keys = _OVERVIEW_STAGED_SETTINGS | set(_enhancement_keys())
        for key in self._image_keys:
            row = self._model._widgets.get(key)
            if row is not None:
                self._listen(row, functools.partial(self._image_row_changed, key))
        browser._tile_size_changed = lambda edge: self._write("overview_tile_size", edge)
        browser._tile_spacing_changed = lambda gap: self._write("overview_tile_spacing", gap)
        browser._write_display_rows = self._write_display_rows
        browser._settings_flush = self.apply
        from spacr.settings import _overview_measurement_declarations

        measurement_keys = tuple(_overview_measurement_declarations()["defaults"])
        browser._measurement_settings = lambda: {
            key: self._value(key) for key in measurement_keys}
        self.apply_image_settings()

    def _listen(self, row, slot):
        """Connect ``slot`` to the row's change signal and remember it for :meth:`_detach`."""
        from ..screens.settings_model import _VALUE_CHANGED_SIGNALS

        for name in _VALUE_CHANGED_SIGNALS:
            signal = getattr(row, name, None)
            if signal is not None:
                signal.connect(slot)
                self._connections.append((signal, slot))
                return

    def _detach(self, *_):
        """Disconnect every setting row from this bridge's browser.

        THE ROWS OUTLIVE THE BROWSER. They belong to the Overview tab's
        settings, while a browser is replaced after each metadata step and
        deleted with its controls; a row edited afterwards called into a
        deleted line edit (GitHub #166). Called when the page retires the
        browser and when the browser is destroyed.
        """
        self._detached = True
        connections, self._connections = self._connections, []
        for signal, slot in connections:
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass

    def _live(self, *controls):
        """Whether this bridge is attached and every control still exists."""
        from shiboken6 import isValid

        if self._detached or not isValid(self._browser) or not all(isValid(c) for c in controls):
            self._detach()
            return False
        return True

    def _row_changed(self, key, control, *_):
        """Follow a changed setting row with its browser control, or hold a display change for Apply."""
        if self._syncing or self._stage(key):
            return
        self._to_control(key, control)

    def _image_row_changed(self, key, *_):
        """Apply a changed Image or Enhancement setting, or hold a display change for Apply."""
        if self._stage(key):
            return
        self.apply_image_settings()

    def _stage(self, key):
        """Hold a changed display or normalization setting for Apply.

        Not held: settings outside :data:`_OVERVIEW_STAGED_SETTINGS` and the
        enhancement settings, values the browser writes itself, and edits
        while a channel is picked in Channel to edit, which wait for that
        channel's own Apply settings to channel.

        :param key: the changed setting.
        :returns: True when the change is held.
        """
        if (key not in self._staged_keys or self._applying or self._syncing or self._writing_display
                or self._detached or self._browser._editing_channel is not None):
            return False
        self._staged.add(key)
        self._show_pending()
        return True

    def pending(self):
        """The settings changed since the last Apply, sorted."""
        return tuple(sorted(self._staged))

    def _show_pending(self):
        """Show on the Apply button whether changes wait, and whether they can be applied."""
        if self.apply_bar is None:
            return
        from shiboken6 import isValid

        if not isValid(self.apply_bar):
            self.apply_bar = None
            return
        error = _overview_channel_list_error(self._value("overview_normalize_channels"),
                                             tuple(self._browser._channel_buttons))
        self.apply_bar.show_state(bool(self._staged), error)

    def apply(self):
        """Apply every held display and normalization change to every view.

        The plate thumbnails, the well views, every open image view and the
        enlarged image are redrawn with the new settings, the image already
        open included.

        :returns: True when the settings were applied; False while Normalized
            channels holds text that is not a channel list.
        """
        if not self._live():
            return False
        error = _overview_channel_list_error(self._value("overview_normalize_channels"),
                                             tuple(self._browser._channel_buttons))
        if error:
            self._show_pending()
            self._browser.status.setText(error)
            return False
        staged, self._staged = set(self._staged), set()
        self._applying = True
        try:
            for key, _row, control in self._pairs:
                if key in staged:
                    self._to_control(key, control)
            self.apply_image_settings()
        finally:
            self._applying = False
        if staged:
            self._browser._apply_display()
        self._show_pending()
        return True

    def _value(self, key):
        """The setting's current value as its row reads it."""
        return self._model._read_value(key)

    def _write(self, key, value):
        """Write ``value`` into the setting row without echoing it back."""
        if self._syncing:
            return
        self._syncing = True
        try:
            self._model.set_value_for_key(key, value)
        finally:
            self._syncing = False

    def _to_control(self, key, control):
        """Follow one setting row with its browser control."""
        if self._syncing or not self._live(control):
            return
        self._syncing = True
        try:
            _overview_set_control(control, self._value(key))
        finally:
            self._syncing = False

    def _to_setting(self, key, control):
        """Write one browser control's value into its setting row."""
        if not self._live(control):
            return
        value = _overview_control_value(control)
        if value is not None:
            self._write(key, value)

    def _write_display_rows(self, values):
        """Show a channel's (or the global) ink intensity and enhancement values in their rows.

        :param values: ``ink`` and ``enhance_*`` values.
        """
        if not self._live():
            return
        self._writing_display = True
        try:
            for name, value in values.items():
                key = "overview_ink_intensity" if name == "ink" else name
                if self._model._widgets.get(key) is not None and self._value(key) != value:
                    self._model.set_value_for_key(key, value)
        finally:
            self._writing_display = False

    def apply_image_settings(self):
        """Apply the Image and Enhancement settings to every view.

        While a channel is picked in Channel to edit, the ink intensity and
        enhancement rows hold that channel's draft instead, applied by Apply
        settings to channel.
        """
        if not self._live():
            return
        browser = self._browser
        values = {key: self._value(key) for key in self._image_keys}
        from spacr.settings import _enhancement_keys

        enhance_values = {key: values.get(key) for key in _enhancement_keys()}
        editing = browser._editing_channel is not None
        if editing:
            browser._channel_display_draft(dict(enhance_values, ink=values.get("overview_ink_intensity")))
        else:
            browser._set_channel_display_globals(values.get("overview_ink_intensity"), enhance_values)
        try:
            if values.get("overview_tile_spacing") is not None:
                browser._set_tile_spacing(int(values["overview_tile_spacing"]))
            if values.get("overview_tile_size") is not None:
                browser._set_tile_size(int(values["overview_tile_size"]))
        except (TypeError, ValueError):
            pass
        browser._set_color_mode(values.get("overview_color_mode"),
                                _overview_custom_colors(values.get("overview_custom_colors")))
        browser._set_load_limits(values.get("overview_max_wells"), values.get("overview_max_fields"))
        browser._set_normalization_scopes(_overview_channel_scopes(values.get("overview_normalization_scope")))
        normalize = values.get("overview_normalize_channels")
        if _overview_channel_list_error(normalize, tuple(browser._channel_buttons)):
            normalize = browser._normalize_channels
        else:
            normalize = _overview_channel_targets(normalize)
        browser._set_channel_targets(_overview_channel_targets(values.get("overview_enhance_channels")),
                                     normalize)
        if editing:
            return
        try:
            chain = _overview_chain(values)
        except ValueError as error:
            browser.status.setText(tr("Image enhancement is not applied: {error}", error=error))
            chain = None
        browser._set_enhancement(chain)


def _overview_row_label(caption, tooltip):
    """A setting-style caption that carries its own tooltip.

    :param caption: the row's English name.
    :param tooltip: the row's English help.
    :returns: the label.
    """
    label = QLabel(tr(caption))
    label.setObjectName("OverviewRowLabel")
    label.setToolTip(tr(tooltip))
    label.setWordWrap(True)
    label.setCursor(Qt.WhatsThisCursor)
    return label


class _OverviewChannelToggles(QWidget):
    """One switch per channel, each on its own line, for a channel-target setting.

    Stands in for the comma-separated text of ``overview_normalize_channels``
    or ``overview_enhance_channels``: the setting stays the record (``"all"``,
    ``"none"`` or ``"1,3"``) and this writes it.
    """

    def __init__(self, model, key, parent=None):
        """Follow one setting row.

        :param model: the Overview tab's settings model.
        :param key: the channel-target setting it writes.
        :param parent: the owning widget.
        """
        from PySide6.QtWidgets import QVBoxLayout
        from ..screens.settings_model import _connect_value_changed

        super().__init__(parent)
        self._model, self._key = model, key
        self._channels = ()
        self._toggles = {}
        self._writing = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self._layout = layout
        row = model._widgets.get(key)
        if row is not None:
            _connect_value_changed(row, self._row_changed)

    def set_channels(self, channels):
        """Show one switch per channel, set from the setting's value.

        :param channels: the source's channel IDs, in order.
        """
        from .toggle import Toggle

        channels = tuple(channels)
        if channels == self._channels:
            self._row_changed()
            return
        for toggle in self._toggles.values():
            self._layout.removeWidget(toggle)
            toggle.deleteLater()
        self._toggles = {}
        self._channels = channels
        for channel in channels:
            toggle = Toggle(tr("Channel {channel}", channel=channel))
            toggle.toggled.connect(self._toggled)
            self._layout.addWidget(toggle)
            self._toggles[channel] = toggle
        self._row_changed()

    def _row_changed(self, *_):
        """Follow the setting row's value."""
        if self._writing:
            return
        targets = _overview_channel_targets(self._model._read_value(self._key))
        for channel, toggle in self._toggles.items():
            toggle.blockSignals(True)
            toggle.setChecked(targets is None or channel in targets or str(channel) in map(str, targets))
            toggle.blockSignals(False)

    def _toggled(self, *_):
        """Write the switched-on channels back into the setting."""
        on = [str(channel) for channel, toggle in self._toggles.items() if toggle.isChecked()]
        value = "all" if len(on) == len(self._toggles) else (",".join(on) or "none")
        self._writing = True
        try:
            self._model.set_value_for_key(self._key, value)
        finally:
            self._writing = False


_OVERVIEW_SCOPE_CHOICES = (("image", "Each image"), ("well", "Each well"),
                           ("plate", "Each plate"), ("all", "All loaded plates"))


class _OverviewChannelScopes(QWidget):
    """One scope choice per channel, each on its own line, for ``overview_normalization_scope``.

    The setting stays the record (``"image"`` or ``"1=plate, 2=image"``) and
    this writes it.
    """

    def __init__(self, model, key, parent=None):
        """Follow one setting row.

        :param model: the Overview tab's settings model.
        :param key: the scope setting it writes.
        :param parent: the owning widget.
        """
        from PySide6.QtWidgets import QFormLayout
        from ..screens.settings_model import _connect_value_changed

        super().__init__(parent)
        self._model, self._key = model, key
        self._channels = ()
        self._choices = {}
        self._writing = False
        self._layout = QFormLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        row = model._widgets.get(key)
        if row is not None:
            _connect_value_changed(row, self._row_changed)

    def set_channels(self, channels):
        """Show one scope choice per channel, set from the setting's value.

        :param channels: the source's channel IDs, in order.
        """
        channels = tuple(channels)
        if channels == self._channels:
            self._row_changed()
            return
        while self._layout.rowCount():
            self._layout.removeRow(0)
        self._choices = {}
        self._channels = channels
        for channel in channels:
            choice = QComboBox()
            for value, caption in _OVERVIEW_SCOPE_CHOICES:
                choice.addItem(tr(caption), value)
            choice.currentIndexChanged.connect(self._chosen)
            self._layout.addRow(tr("Channel {channel}", channel=channel), choice)
            self._choices[channel] = choice
        self._row_changed()

    def _row_changed(self, *_):
        """Follow the setting row's value."""
        if self._writing:
            return
        scopes = _overview_channel_scopes(self._model._read_value(self._key))
        for channel, choice in self._choices.items():
            choice.blockSignals(True)
            choice.setCurrentIndex(max(0, choice.findData(_overview_scope_of(scopes, channel))))
            choice.blockSignals(False)

    def _chosen(self, *_):
        """Write the chosen scopes back into the setting."""
        value = _overview_scope_text({str(channel): choice.currentData() for channel, choice in self._choices.items()})
        self._writing = True
        try:
            self._model.set_value_for_key(self._key, value)
        finally:
            self._writing = False


_OVERVIEW_CUSTOM_COLORS_ROW = (
    "Custom colours",
    "The colour of each channel in the Custom colours colour mode; click a channel's button to pick "
    "its colour. Saved with the other settings.",
)


class _OverviewCustomColors(QWidget):
    """One colour button per channel for ``overview_custom_colors``, used by the custom colour mode.

    The setting stays the record (``"1=#ff0000, 2=#00ff00"``) and this
    writes it; the buttons are greyed out unless the colour mode is custom.
    """

    def __init__(self, model, key, parent=None):
        """Follow one setting row and the colour mode.

        :param model: the Overview tab's settings model.
        :param key: the custom-colour setting it writes.
        :param parent: the owning widget.
        """
        from PySide6.QtWidgets import QFormLayout
        from ..screens.settings_model import _connect_value_changed

        super().__init__(parent)
        self._model, self._key = model, key
        self._channels = ()
        self._buttons = {}
        self._writing = False
        self._layout = QFormLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        for name, slot in ((key, self._row_changed), ("overview_color_mode", self._mode_changed)):
            row = model._widgets.get(name)
            if row is not None:
                _connect_value_changed(row, slot)
        self._mode_changed()

    def colours(self):
        """``{channel text: '#rrggbb'}`` as the setting holds them, defaults filled in."""
        own = _overview_custom_colors(self._model._read_value(self._key))
        return {str(channel): _overview_custom_color(own, channel, index)
                for index, channel in enumerate(self._channels)}

    def set_channels(self, channels):
        """Show one colour button per channel.

        :param channels: the source's channel IDs, in order.
        """
        channels = tuple(channels)
        if channels == self._channels:
            self._row_changed()
            return
        while self._layout.rowCount():
            self._layout.removeRow(0)
        self._buttons = {}
        self._channels = channels
        for channel in channels:
            button = QPushButton()
            button.setObjectName("OverviewCustomColour")
            button.setMaximumWidth(160)
            button.setToolTip(tr("Pick the colour of channel {channel}.", channel=channel))
            button.clicked.connect(functools.partial(self._pick, channel))
            self._layout.addRow(tr("Channel {channel}", channel=channel), button)
            self._buttons[str(channel)] = button
        self._row_changed()
        self._mode_changed()

    def _row_changed(self, *_):
        """Show each channel's colour on its button."""
        for channel, colour in self.colours().items():
            button = self._buttons.get(channel)
            if button is not None:
                button.setText(colour)
                button.setStyleSheet(f"QPushButton#OverviewCustomColour {{ border-left: 18px solid {colour}; }}")

    def _mode_changed(self, *_):
        """Grey the buttons out unless the colour mode is custom."""
        custom = str(self._model._read_value("overview_color_mode") or "") == "custom"
        self.setEnabled(custom)
        for channel, button in self._buttons.items():
            button.setToolTip(tr("Pick the colour of channel {channel}.", channel=channel) if custom
                              else tr("Used only when the colour mode is Custom colours."))

    def _pick(self, channel, *_):
        """Ask for one channel's colour with spaCR's colour picker and write it into the setting."""
        from .colour_picker import pick_colour

        colours = self.colours()
        chosen = pick_colour(self, colours.get(str(channel)), tr("Colour of channel {channel}", channel=channel))
        if not chosen.isValid():
            return
        self.set_colour(channel, chosen.name())

    def set_colour(self, channel, colour):
        """Give ``channel`` the colour ``'#rrggbb'`` and save it in the setting."""
        colours = self.colours()
        colours[str(channel)] = str(colour).lower()
        self._writing = True
        try:
            self._model.set_value_for_key(self._key, _overview_custom_text(colours))
        finally:
            self._writing = False
        self._row_changed()


class _OverviewApplyBar(QFrame):
    """The Apply button pinned to the bottom of the Overview's settings column.

    Display and normalization settings wait for it; while some do, it says
    so and the button is highlighted. While Normalized channels holds text
    that is not a channel list it says why and cannot be pressed.
    """

    def __init__(self, scroll):
        """Build the bar on the settings column's scroll area.

        :param scroll: the settings column; the bar sits over its bottom edge.
        """
        super().__init__(scroll)
        self.setObjectName("OverviewApplyBar")
        self.setAutoFillBackground(True)
        self._scroll = scroll
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        self.state = QLabel(tr("Display settings are applied."))
        self.state.setObjectName("OverviewApplyState")
        self.state.setWordWrap(True)
        self.apply_button = QPushButton(tr("Apply"))
        self.apply_button.setObjectName("OverviewApplySettings")
        self.apply_button.setToolTip(tr(
            "Apply the changed channel, normalization, colour and enhancement settings to the plate, "
            "every well view and every open image."))
        layout.addWidget(self.state, 1)
        layout.addWidget(self.apply_button)
        from ..theme import ensure_widget_qss_applied, register_widget_qss

        register_widget_qss("OverviewApplyBar", _overview_apply_bar_qss, replace=True)
        ensure_widget_qss_applied("OverviewApplyBar", root=self)

    def show_state(self, pending, error=""):
        """Show whether changes wait for Apply, or what stops them being applied.

        :param pending: whether display changes wait.
        :param error: why they cannot be applied, or ``""``.
        """
        if error:
            text = error
        elif pending:
            text = tr("Unapplied changes: press Apply to show them in every view.")
        else:
            text = tr("Display settings are applied.")
        self.state.setText(text)
        if bool(self.state.property("error")) != bool(error):
            self.state.setProperty("error", bool(error))
            self.state.style().unpolish(self.state)
            self.state.style().polish(self.state)
        self.apply_button.setEnabled(not error)
        self.apply_button.setProperty("pending", bool(pending))
        self.apply_button.setText(tr("Apply changes") if pending else tr("Apply"))
        self.apply_button.style().unpolish(self.apply_button)
        self.apply_button.style().polish(self.apply_button)
        self.place()

    def place(self):
        """Sit over the bottom of the settings column and keep the settings scrollable above it."""
        scroll = self._scroll
        height = self.sizeHint().height()
        frame = scroll.frameWidth()
        scroll.setViewportMargins(0, 0, 0, height)
        self.setGeometry(frame, scroll.height() - frame - height, max(0, scroll.width() - 2 * frame), height)
        self.raise_()

    def eventFilter(self, watched, event):
        """Follow the settings column's size."""
        from PySide6.QtCore import QEvent

        if watched is self._scroll and event.type() in (QEvent.Resize, QEvent.Show):
            self.place()
        return False


def _overview_apply_bar_qss(palette, opacity):
    """The Apply bar: the settings surface with a rule above it, and an accent Apply while changes wait."""
    return (f"QFrame#OverviewApplyBar {{ background-color: {palette['surface']}; "
            f"border-top: 1px solid {palette['border']}; }}\n"
            f"QPushButton#OverviewApplySettings[pending=\"true\"] {{ background-color: {palette['accent']}; "
            f"color: {palette['bg']}; border: 1px solid {palette['accent']}; font-weight: bold; }}\n"
            f"QLabel#OverviewApplyState {{ color: {palette['fg']}; }}\n"
            f"QLabel#OverviewApplyState[error=\"true\"] {{ color: {palette['error']}; }}")


def _overview_channel_list_qss(palette, opacity):
    """A Normalized channels field that is not a channel list is framed in the error colour."""
    return f"QLineEdit[overviewInvalid=\"true\"] {{ border: 1px solid {palette['error']}; }}"


def _overview_rows_hidden(previous, keys):
    """The rows the screen hides, plus the text rows the channel switches stand in for.

    :param previous: what the screen hid before, a callable or ``None``.
    :param keys: the channel-target settings shown as switches.
    :returns: the keys to keep off the form.
    """
    return set(previous() or ()) | set(keys) if previous is not None else set(keys)


_OVERVIEW_CHANNEL_TARGET_ROWS = (
    ("Enhancement", "overview_enhance_channels", "Enhanced channels",
     "Channels the image enhancement applies to; a channel switched off is shown without enhancement."),
)

_OVERVIEW_NORMALIZATION_SCOPE_ROW = (
    "Normalization scope",
    "Per channel, which images share one display stretch: each image on its own, each well, "
    "each plate, or all loaded plates. Wider scopes use sampled pixels: at most 64 images "
    "per scope and 4096 pixels per image.",
)


def _overview_row_key(caption):
    """The help key of a mounted Overview row, such as ``overview_plate`` for "Plate".

    :param caption: the row's English name.
    :returns: the key its popup help is filed under.
    """
    return "overview_" + "_".join(caption.lower().split())


def _overview_row_field(widgets):
    """Put one or more controls side by side as a row's field.

    :param widgets: the controls.
    :returns: the field holding them.
    """
    from .flow import FlowHost, FlowLayout

    if len(widgets) == 1:
        widgets[0].show()
        return widgets[0]
    host = FlowHost()
    flow = FlowLayout(host)
    for widget in widgets:
        widget.show()
        flow.addWidget(widget)
    return host


def _overview_module_rows(browser):
    """The browser controls that are not settings, by category, as rows.

    :param browser: the plate browser.
    :returns: ``{category: [(caption, tooltip, widgets), ...]}``.
    """
    return {
        "Figure settings": [
            ("Plate", "Plate shown in the plate view.", (browser.plates,)),
            ("Row", "Restrict the plate view to one row, or show every row.", (browser.rows,)),
            ("Column", "Restrict the plate view to one column, or show every column.", (browser.columns,)),
            ("Field", "Field of view shown for each well.", (browser.fov_axis,)),
            ("Time", "Time point shown for time-lapse acquisitions.", (browser.time_axis,)),
            ("Z", "Z plane shown when the Z mode is a single plane.", (browser.z_axis,)),
        ],
        "Advanced normalization": [
            ("Reference group", "Group whose images define the advanced normalization bounds.", (browser.normalization_scope,)),
            ("Channel to edit", "Channel the per-channel settings are edited for: the advanced normalization, the percentile stretch, every enhancement step and the CMY/CMYK ink intensity. Pick a channel, change its values and press Apply settings to channel; channels marked (own settings) have their own values, the others follow All channels.",
             (browser.norm_channel, *browser._channel_override_buttons)),
        ],
        "Comparisons": [
            ("Annotation dimensions", "Tick annotation columns and the values to compare; every combination of the ticked values is one group.",
             (browser.comparison_dimensions,)),
            ("Group values", "Value, or Any, of each annotation column in the next group; Any pools that column.",
             (browser.group_builder.picker_host,)),
            ("Groups", "Groups compared, in order, with their labels and image counts; add, remove and reorder them here.",
             (browser.group_builder.group_host,)),
            ("Group presets", "Fill the groups with every combination of the ticked values, or vary one column and hold the others.",
             (browser.group_builder.preset_host,)),
            ("Compare within", "Compare groups separately within each value of this annotation column.", (browser.comparison_within,)),
            ("Measurement", "Image-level measurement to compare; leave empty to use the first measured column.", (browser.comparison_measurement,)),
            ("Comparison actions", "Draw the group comparison figure and tables, or show one representative image per group.",
             browser._comparison_buttons),
        ],
        "Export": [
            ("Selection", "Clear every selected field and well.", (browser._clear_selection_button,)),
        ],
    }


def _overview_section(screen, title):
    """The module's built heading for category ``title``, opening it if it waits.

    :param screen: the Overview module screen.
    :param title: the category title.
    :returns: the heading, or ``None``.
    """
    from ..screens.settings_model import categories_for_app, get_categories

    keys = categories_for_app(screen.app_key, get_categories()).get(title, ())
    return screen._heading_of_category(title, keys)


class _OverviewModulePage(QWidget):
    """The Overview module's image area: the metadata step, then the browser."""

    def __init__(self, screen):
        """Build an empty page that waits for Run.

        :param screen: the Overview module screen that owns the settings.
        """
        from PySide6.QtWidgets import QStackedWidget

        super().__init__(screen)
        self._screen = screen
        self.wizard = None
        self.browser = None
        self._bridge = None
        self._rows = []
        self._hidden_keys = []
        self._screen_hides = None
        self._apply_bar = None
        self._channel_list_edit = None
        self.setObjectName("FieldOverviewModulePage")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        navigation = QHBoxLayout()
        self.back_button = QPushButton(tr("Back"))
        self.back_button.clicked.connect(self._back)
        self.back_button.hide()
        navigation.addWidget(self.back_button)
        navigation.addStretch(1)
        layout.addLayout(navigation)
        self.stack = QStackedWidget(self)
        self.placeholder = QLabel(tr("Choose an image folder as the source and press Run to open its plate overview."))
        self.placeholder.setWordWrap(True)
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.stack.addWidget(self.placeholder)
        self.stack.currentChanged.connect(self._sync_back)
        self.stack.currentChanged.connect(self._sync_categories)
        layout.addWidget(self.stack, 1)
        from ..theme import ensure_widget_qss_applied, register_widget_qss

        register_widget_qss("FieldOverviewModulePage", _overview_module_qss, replace=True)
        ensure_widget_qss_applied("FieldOverviewModulePage", root=self)
        self._mount_full_resolution_steps()

    def _mount_full_resolution_steps(self):
        """Offer Make Masks' PSF and Cellpose 3 controls under Enhancement.

        They build their kernel and load their model off the GUI thread and
        apply only to images opened with Original.
        """
        from .psf_controls import _PSFControls
        from .restoration_controls import _RestorationControls

        self.psf_controls = _PSFControls(image_paths=self._current_image_paths)
        self.restoration_controls = _RestorationControls()
        self.psf_controls.changed.connect(self._full_resolution_changed)
        self.restoration_controls.changed.connect(self._full_resolution_changed)
        console = getattr(self._screen, "_console", None)
        if console is not None:
            self.restoration_controls.said.connect(lambda text, _kind: console.append_stdout(text + "\n"))
        section = _overview_section(self._screen, "Enhancement")
        if section is None:
            return
        note = QLabel(tr("PSF deconvolution and Cellpose 3 restoration apply only to images opened "
                         "with Original, never to previews or thumbnails."))
        note.setWordWrap(True)
        section.add_prose(note)
        for caption, tooltip, control in (
                ("PSF deconvolution", "Make Masks' calibrated or measured point spread function, applied "
                 "to Original images after background subtraction.", self.psf_controls),
                ("Cellpose 3 restoration", "Make Masks' Cellpose 3 denoising, deblurring or one-click "
                 "restoration, applied to Original images; the model is installed and loaded through "
                 "the Model Zoo.", self.restoration_controls)):
            section.add_prose_row(_overview_row_label(caption, tooltip), control)

    def _current_image_paths(self):
        """The current field's files, for inferring PSF optics from their metadata."""
        browser = self.browser
        item = browser.fields.currentItem() if browser is not None else None
        field = item.data(Qt.UserRole) if item is not None else (browser._fields[0] if browser and browser._fields else None)
        return [path for _channel, path in field.files] if field is not None else []

    def _full_resolution_fields(self):
        """The PSF and restoration chain fields the controls ask for."""
        return {**self.psf_controls._chain_fields(), **self.restoration_controls._chain_fields()}

    def _full_resolution_changed(self):
        """Reload the browser's Original images with the new kernel or model."""
        if self.browser is not None:
            self.browser._original_steps_changed()

    def run(self, settings):
        """Open the metadata step for the module's current source.

        :param settings: the module's collected settings.
        :returns: the metadata step.
        """
        settings = dict(settings or {})
        source = _overview_source_text(settings.get("src"))
        if source in ("", os.path.abspath("path")):
            source = ""
        wizard = self.wizard
        if wizard is not None and not wizard._finished and wizard.source.text() == source:
            self.stack.setCurrentWidget(wizard)
            return wizard
        if wizard is not None:
            self._drop_wizard(wizard)
        channels = [value.strip() for value in str(settings.get("overview_channels") or "").split(",")
                    if value.strip()]
        channels = [int(value) if value.isdigit() else value for value in channels]
        wizard = _OverviewMetadataWizard(source, {
            "src": source, "custom_regex": settings.get("custom_regex") or "",
            "channels": channels, "normalize": bool(settings.get("overview_normalize", True)),
            "normalization_percentiles": (float(settings.get("overview_percentile_low", 1.)),
                                          float(settings.get("overview_percentile_high", 99.9)))}, self)
        wizard._startup_timer.stop()
        wizard.setWindowFlags(Qt.Widget)
        wizard.setObjectName("FieldOverviewMetadata")
        wizard.setAutoFillBackground(False)
        wizard.accepted.connect(lambda owner=wizard: self._show_browser(owner))
        wizard.rejected.connect(lambda owner=wizard: self._wizard_closed(owner))
        self.stack.addWidget(wizard)
        self.stack.setCurrentWidget(wizard)
        self.wizard = wizard
        wizard._startup_timer.start(0)
        return wizard

    def _drop_wizard(self, wizard):
        """Retire a metadata step that is no longer shown."""
        if wizard is self.wizard:
            self.wizard = None
        if not wizard._finished:
            wizard.blockSignals(True)
            wizard.reject()
        self.stack.removeWidget(wizard)
        wizard.deleteLater()

    def _wizard_closed(self, wizard):
        """Return to the previous browser, or the empty page, after Cancel."""
        self._drop_wizard(wizard)
        self.stack.setCurrentWidget(self.browser or self.placeholder)

    def _show_browser(self, wizard):
        """Replace the metadata step with the plate browser it configured."""
        config = wizard.configuration()
        fields = tuple(wizard._fields)
        self._retire_browser()
        browser = _FieldOverview(self, config)
        browser.setWindowFlags(Qt.Widget)
        browser.setObjectName("FieldOverviewBody")
        browser.setAutoFillBackground(False)
        browser.options_button.hide()
        browser._essential_scroll.hide()
        self.browser = browser
        browser._original_steps = self._full_resolution_fields
        self._screen._field_overview = browser
        self.stack.addWidget(browser)
        self.stack.setCurrentWidget(browser)
        browser._mount_navigation_bar()
        self._mount_rows(browser)
        self._bridge = _OverviewSettingsBridge(self._screen, browser)
        self._mount_apply_bar()
        self._drop_wizard(wizard)
        browser._indexed(fields)
        self._sync_back()
        _overview_open_settings(self._screen)

    def _mount_rows(self, browser):
        """Lay the browser's navigation and action controls out as setting rows.

        Each row's help is the module's standard popup -- name, wrapped
        description and API link -- the same as every setting row's.
        """
        from ..screens.settings_model import install_api_tooltips

        keys = {}
        for title, rows in _overview_module_rows(browser).items():
            section = _overview_section(self._screen, title)
            if section is None:
                continue
            for caption, tooltip, widgets in rows:
                label = _overview_row_label(caption, tooltip)
                field = _overview_row_field(widgets)
                key = _overview_row_key(caption)
                field.setProperty("settingKey", key)
                field.setProperty("apiTooltipDescriptionSource", tooltip)
                section.add_prose_row(label, field)
                keys[field] = key
                self._rows.append((section, field, tuple(widgets)))
                for widget in widgets:
                    widget.destroyed.connect(browser._mounted_control_lost)
        model = self._screen._settings_model
        for title, key, caption, tooltip in _OVERVIEW_CHANNEL_TARGET_ROWS:
            section = _overview_section(self._screen, title)
            if section is None or model._widgets.get(key) is None:
                continue
            toggles = _OverviewChannelToggles(model, key)
            toggles.setProperty("settingKey", _overview_row_key(caption))
            toggles.setProperty("apiTooltipDescriptionSource", tooltip)
            browser._channels_found.connect(toggles.set_channels)
            if browser._channel_buttons:
                toggles.set_channels(tuple(browser._channel_buttons))
            section.add_prose_row(_overview_row_label(caption, tooltip), toggles)
            keys[toggles] = _overview_row_key(caption)
            self._rows.append((section, toggles, ()))
            self._hidden_keys.append(key)
        section = _overview_section(self._screen, "Advanced normalization")
        if section is not None and model._widgets.get("overview_normalization_scope") is not None:
            caption, tooltip = _OVERVIEW_NORMALIZATION_SCOPE_ROW
            scopes = _OverviewChannelScopes(model, "overview_normalization_scope")
            scopes.setProperty("settingKey", _overview_row_key(caption))
            scopes.setProperty("apiTooltipDescriptionSource", tooltip)
            browser._channels_found.connect(scopes.set_channels)
            if browser._channel_buttons:
                scopes.set_channels(tuple(browser._channel_buttons))
            section.add_prose_row(_overview_row_label(caption, tooltip), scopes)
            keys[scopes] = _overview_row_key(caption)
            self._rows.append((section, scopes, ()))
            self._hidden_keys.append("overview_normalization_scope")
        section = _overview_section(self._screen, "Channels")
        if section is not None and model._widgets.get("overview_custom_colors") is not None:
            caption, tooltip = _OVERVIEW_CUSTOM_COLORS_ROW
            colours = _OverviewCustomColors(model, "overview_custom_colors")
            colours.setProperty("settingKey", _overview_row_key(caption))
            colours.setProperty("apiTooltipDescriptionSource", tooltip)
            browser._channels_found.connect(colours.set_channels)
            if browser._channel_buttons:
                colours.set_channels(tuple(browser._channel_buttons))
            section.add_prose_row(_overview_row_label(caption, tooltip), colours)
            keys[colours] = _overview_row_key(caption)
            self._rows.append((section, colours, ()))
            self._hidden_keys.append("overview_custom_colors")
        from spacr.settings import _OVERVIEW_WELL_VIEW_KEYS

        self._hidden_keys.extend(key for key in _OVERVIEW_WELL_VIEW_KEYS if model._widgets.get(key) is not None)
        self._mount_channel_list_check(browser, model)
        if self._hidden_keys:
            self._screen_hides = model.rows_the_screen_hides
            model.rows_the_screen_hides = functools.partial(
                _overview_rows_hidden, self._screen_hides, tuple(self._hidden_keys))
            model.refresh_object_visibility()
        objects = model._widgets.get("overview_objects_path")
        if objects is not None:
            from ..screens.settings_model import _add_folder_browse_action

            _add_folder_browse_action(objects, functools.partial(_overview_choose_object_masks, objects, self))
        install_api_tooltips(self._screen, "overview", keys)
        section = _overview_section(self._screen, "Export")
        if section is not None:
            from ..widgets.section import Section

            _overview_height_grip(section)

    def _mount_channel_list_check(self, browser, model):
        """Check Normalized channels as it is typed; a wrong list is framed and the Apply bar says why.

        :param browser: the plate browser, whose channels the list must name.
        :param model: the Overview tab's settings model.
        """
        from PySide6.QtWidgets import QLineEdit
        from ..theme import ensure_widget_qss_applied, register_widget_qss

        edit = model._widgets.get("overview_normalize_channels")
        edits = [edit] if isinstance(edit, QLineEdit) else (edit.findChildren(QLineEdit) if edit is not None else [])
        if not edits:
            return
        edit = edits[0]
        register_widget_qss("OverviewChannelList", _overview_channel_list_qss, replace=True)
        ensure_widget_qss_applied("OverviewChannelList", root=edit)
        self._channel_list_edit = edit
        edit.textChanged.connect(self._check_channel_list)
        browser._channels_found.connect(self._check_channel_list)
        self._check_channel_list()

    def _check_channel_list(self, *_):
        """Frame Normalized channels when it does not name channels of this source and say why.

        :returns: the reason, or ``""`` when the list is valid.
        """
        from shiboken6 import isValid

        edit = self._channel_list_edit
        if edit is None or not isValid(edit):
            return ""
        browser = self.browser
        known = tuple(browser._channel_buttons) if browser is not None else ()
        error = _overview_channel_list_error(edit.text(), known)
        if bool(edit.property("overviewInvalid")) != bool(error):
            edit.setProperty("overviewInvalid", bool(error))
            edit.style().unpolish(edit)
            edit.style().polish(edit)
        if error and browser is not None:
            browser.status.setText(error)
        if self._bridge is not None:
            self._bridge._show_pending()
        return error

    def _drop_channel_list_check(self):
        """Stop checking Normalized channels and take its error frame off."""
        from shiboken6 import isValid

        edit, self._channel_list_edit = self._channel_list_edit, None
        if edit is None or not isValid(edit):
            return
        try:
            edit.textChanged.disconnect(self._check_channel_list)
        except (RuntimeError, TypeError):
            pass
        edit.setProperty("overviewInvalid", False)
        edit.style().unpolish(edit)
        edit.style().polish(edit)

    def _mount_apply_bar(self):
        """Pin the Apply button to the bottom of the settings column while the browser is shown."""
        scroll = getattr(self._screen, "_settings_scroll", None)
        if scroll is None or self._bridge is None:
            return
        bar = _OverviewApplyBar(scroll)
        bar.apply_button.clicked.connect(self._apply_settings)
        self._bridge.apply_bar = bar
        self._apply_bar = bar
        scroll.installEventFilter(bar)
        bar.place()
        bar.show()
        self._bridge._show_pending()

    def _apply_settings(self, *_):
        """Apply the held display and normalization settings to every view."""
        if self._bridge is not None:
            self._bridge.apply()

    def _drop_apply_bar(self):
        """Take the Apply button off the settings column."""
        from shiboken6 import isValid

        bar, self._apply_bar = getattr(self, "_apply_bar", None), None
        if bar is None or not isValid(bar):
            return
        scroll = bar.parentWidget()
        if scroll is not None:
            scroll.removeEventFilter(bar)
            scroll.setViewportMargins(0, 0, 0, 0)
        bar.hide()
        bar.deleteLater()

    def _retire_browser(self):
        """Retire the shown browser and take its rows off the settings.

        The bridge is detached first, and the browser's own controls go back
        under the browser before their rows are removed, so they are deleted
        with it and never before it (GitHub #165, #166). Each row is found
        while its field is still in the form and removed by its index: moving
        a lone control out takes it from the form, which left the row's label
        behind and made ``removeRow(field)`` warn ``Invalid widget``.
        """
        from shiboken6 import isValid

        browser, bridge = self.browser, self._bridge
        self.browser, self._bridge = None, None
        self._drop_apply_bar()
        self._drop_channel_list_check()
        if bridge is not None:
            bridge._detach()
        if browser is not None:
            browser._retire()
            self.stack.removeWidget(browser)
        for section, field, widgets in self._rows:
            try:
                row = section._form.getWidgetPosition(field)[0] if isValid(field) else -1
            except RuntimeError:
                row = -1
            if browser is not None and isValid(browser):
                for widget in widgets:
                    if isValid(widget):
                        try:
                            widget.destroyed.disconnect(browser._mounted_control_lost)
                        except (RuntimeError, TypeError):
                            pass
                        widget.hide()
                        widget.setParent(browser)
            if row >= 0:
                try:
                    section._form.removeRow(row)
                except RuntimeError:
                    pass
        self._rows.clear()
        model = self._screen._settings_model
        if self._hidden_keys:
            model.rows_the_screen_hides = self._screen_hides
            self._hidden_keys.clear()
            model.refresh_object_visibility()
        if browser is not None:
            browser.deleteLater()

    def _held_back_sections(self):
        """``id()`` of the settings headings held back until Next opens the browser.

        Before Run only Source is on the form, since it names what Run
        opens; while the metadata step is shown it carries the settings that
        apply itself, so every category stays off the form.
        """
        current = self.stack.currentWidget()
        if self.browser is not None and current is self.browser:
            return set()
        keep = "Source" if current is self.placeholder else None
        return {id(section) for section in self._screen.rendered_settings_sections()
                if str(section.property("settingsCategorySource") or "") != keep}

    def _metadata_step_shown(self):
        """Whether the metadata step, before its Next, fills the page."""
        return self.wizard is not None and self.stack.currentWidget() is self.wizard

    def _sync_categories(self, *_):
        """Show the categories and the settings column only once the browser is shown."""
        refresh = getattr(self._screen, "refresh_maturity_visibility", None)
        if callable(refresh):
            refresh()
        _overview_open_settings(self._screen)

    def _sync_back(self, *_):
        """Show Back only while a well or an image is open."""
        browser = self.browser
        self.back_button.setVisible(
            browser is not None and self.stack.currentWidget() is browser
            and (browser._individual or browser._scope[1] is not None))

    def _back(self):
        """Return from an image to its well, and from a well to the plate."""
        browser = self.browser
        if browser is None:
            return
        if browser._individual:
            browser._individual = False
            browser._comparison_signature = None
            browser._filter()
        elif browser._scope[1] is not None:
            browser._back()
        self._sync_back()

    def stop(self):
        """Cancel the metadata scan or the browser's loading and exports."""
        if self.wizard is not None and not self.wizard._finished:
            self.wizard._cancelled.set()
        if self.browser is not None:
            self.browser._cancel()

    def export_pdfs(self):
        """Write contact sheet PDFs of the current browser's export scope."""
        if self.browser is not None:
            self.browser._export()

    def export_menu(self, anchor=None):
        """Offer the browser's exports under ``anchor``; returns the menu, or ``None`` before Run."""
        if self.browser is None:
            return None
        return self.browser._export_menu(anchor)

    def export_measurements(self):
        """Export image-level measurements of the current browser."""
        if self.browser is not None:
            self.browser._measure()

    def retire(self):
        """Retire the metadata step, the browser and the full-resolution controls."""
        if self.wizard is not None:
            self._drop_wizard(self.wizard)
        self._retire_browser()
        self.psf_controls._shutdown()
        self.restoration_controls._shutdown()


def _overview_module_qss(palette, opacity):
    """Let the Overview module page show the module's own backdrop."""
    return ("QWidget#FieldOverviewModulePage, QWidget#FieldOverviewBody, "
            "QWidget#FieldOverviewMetadata { background: transparent; }\n"
            + _overview_regex_qss(palette))


class _OverviewHeightGrip(QFrame):
    """Drag handle that makes a settings category taller or shorter.

    Dragging adds height below the category's rows; double-clicking gives the
    height back to its content.
    """

    HEIGHT = 7

    def __init__(self, section):
        """Attach the handle to the bottom of ``section``'s body.

        :param section: the settings category it resizes.
        """
        super().__init__(section)
        self._section = section
        self._press_y = None
        self._start = 0
        self.setObjectName("OverviewCategoryHeightGrip")
        self.setCursor(Qt.SizeVerCursor)
        self.setFixedHeight(self.HEIGHT)
        self.setToolTip(tr("Drag to make this category taller or shorter; double-click to fit its settings."))

    def _body(self):
        """The category body whose height the handle sets."""
        return self._section._body

    def mousePressEvent(self, event):
        """Remember where the drag started, and from what height."""
        if event.button() == Qt.LeftButton:
            self._press_y = event.globalPosition().y()
            self._start = self._body().height()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Resize by how far the pointer moved since the press."""
        if self._press_y is not None and event.buttons() & Qt.LeftButton:
            self.set_height(self._start + int(event.globalPosition().y() - self._press_y))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """End the drag."""
        self._press_y = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Give the height back to the category's content."""
        if event.button() == Qt.LeftButton:
            self.set_height(0)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def set_height(self, height):
        """Give the category body at least ``height`` pixels.

        :param height: the wanted body height; 0 fits the rows.
        """
        body = self._body()
        body.setMinimumHeight(max(0, int(height)))
        body.updateGeometry()


def _overview_height_grip(section):
    """Give a settings category a drag handle along its bottom edge, once.

    :param section: the category.
    :returns: the handle.
    """
    grip = getattr(section, "_overview_height_grip", None)
    if grip is None:
        grip = _OverviewHeightGrip(section)
        section.layout().addWidget(grip)
        section._body.setVisible(section.is_expanded())
        grip.setVisible(section.is_expanded())
        section.toggled.connect(grip.setVisible)
        section._overview_height_grip = grip
    return grip


def _open_overview_module(owner, settings=None):
    """Open the Overview as a closable tab beside ``owner``'s own page.

    The owner's settings and view stay on their own tab; closing the Overview
    tab keeps its screen, so the same icon puts it back as it was left.

    :param owner: the module asking, such as Mask generation or Make Masks.
    :param settings: settings naming the source; ``src`` is copied over.
    :returns: the Overview's metadata step.
    """
    from ..screens.map_barcodes import show_as_page

    settings = dict(settings or {})
    tab = getattr(owner, "_overview_tab", None)
    if tab is None:
        from ..screens.app_screen import AppScreen

        tab = AppScreen("overview")
        tab._header.hide()
        owner._overview_tab = tab
    if show_as_page(tab, owner, tr("Overview")) is None:
        return _open_field_overview(owner, settings)
    source = settings.get("src") or settings.get("source")
    model = tab._settings_model
    if source:
        model.set_value_for_key("src", _overview_source_text(source))
    percentiles = settings.get("normalization_percentiles")
    if isinstance(percentiles, (list, tuple)) and len(percentiles) == 2:
        model.set_value_for_key("overview_percentile_low", float(percentiles[0]))
        model.set_value_for_key("overview_percentile_high", float(percentiles[1]))
    wizard = tab._overview_page.run(model.collect())
    tab._overview_page._sync_categories()
    return wizard


_OVERVIEW_SETTINGS_WIDTH = 520


def _overview_choose_object_masks(edit, page):
    """Pick a folder of Mask generation object masks for the Overview's object measurements.

    The folder is checked with :func:`spacr.measure._validate_object_masks`
    before it is accepted; a refusal is shown with its reason and the setting
    keeps its value.

    :param edit: the ``overview_objects_path`` field.
    :param page: the Overview page, whose current browser shows the result.
    :returns: the validation result, or ``None`` when nothing was accepted.
    """
    from PySide6.QtWidgets import QFileDialog
    from ...measure import _validate_object_masks

    path = QFileDialog.getExistingDirectory(edit, tr("Choose object masks…"), edit.text())
    if not path:
        return None
    status = page.browser.status if page.browser is not None else None
    try:
        found = _validate_object_masks(path)
    except ValueError as error:
        if status is not None:
            status.setText(tr("These object masks cannot be used: {error}", error=str(error)))
        return None
    edit.setText(path)
    edit.editingFinished.emit()
    if status is not None:
        status.setText(tr("Object masks: {count} files of {objects}.", count=found["count"],
                          objects=", ".join(sorted(found["roles"]))))
    return found


def _overview_open_settings(screen):
    """Open the Overview tab's settings column, at least :data:`_OVERVIEW_SETTINGS_WIDTH` wide.

    The figures card is a focus pane, so showing the plate folded the
    settings away to the left -- on opening the tab and again after Next.
    Opening it as the user would pins it open for the visit. Before Next
    its categories are held back, so the empty column is folded instead
    and the metadata step, with its own settings, has the width.

    :param screen: the Overview module screen.
    """
    from .collapsible_splitter import CollapsibleSplitter

    body = getattr(screen, "_body_splitter", None)
    if not isinstance(body, CollapsibleSplitter) or body.pane("Settings") is None:
        return
    page = getattr(screen, "_overview_page", None)
    if page is not None and page._metadata_step_shown():
        body.set_collapsed("Settings", True)
        return
    body.set_collapsed("Settings", False, by_user=True)
    sizes = body.sizes()
    total = sum(sizes)
    if len(sizes) == 2 and total > 0:
        wide = min(max(sizes[0], _OVERVIEW_SETTINGS_WIDTH), total // 2)
        if wide != sizes[0]:
            body.setSizes([wide, total - wide])


def _install_overview_module(screen, layout):
    """Lay out the Overview module's image area and console.

    :param screen: the Overview module screen being built.
    :param layout: the runtime column's layout.
    :returns: the splitter holding the image area above the console.
    """
    splitter = screen._new_runtime_splitter()
    splitter.setChildrenCollapsible(False)
    page = _OverviewModulePage(screen)
    screen._overview_page = page
    page._sync_categories()
    screen._figure_queue.hide()
    screen._figures_card.body_layout.addWidget(page, 1)
    screen._figures_card.show()
    screen._figures_card.setMinimumHeight(360)
    splitter.addWidget(screen._figures_card)
    splitter.addWidget(screen._console_wrap)
    splitter.setStretchFactor(0, 4)
    splitter.setStretchFactor(1, 1)
    splitter.setSizes([700, 200])
    layout.addWidget(splitter, 1)
    return splitter


def _install_overview_actions(screen, buttons):
    """Add the Overview actions beside the module's standard ones: Measure first, then Export.

    :param screen: the Overview module screen.
    :param buttons: the Actions container's button strip.
    """
    screen._btn_overview_pdfs = QPushButton(tr("Export…"))
    screen._btn_overview_pdfs.setObjectName("OverviewExportPdfs")
    screen._btn_overview_pdfs.setToolTip(tr("Guided export, Custom export or contact sheet PDFs, written to the results folder."))
    screen._btn_overview_pdfs.clicked.connect(lambda: screen._overview_page.export_menu(screen._btn_overview_pdfs))
    screen._btn_overview_measure = QPushButton(tr("Export image-level measurements"))
    screen._btn_overview_measure.setObjectName("OverviewExportMeasurements")
    screen._btn_overview_measure.setToolTip(tr(
        "Measure every image in parallel and write the results to the fov table of measurements.db."))
    screen._btn_overview_measure.clicked.connect(lambda: screen._overview_page.export_measurements())
    buttons.addWidget(screen._btn_overview_measure)
    buttons.addWidget(screen._btn_overview_pdfs)


def _overview_source_path(source):
    """Normalize the first source lexically without filesystem work on the GUI thread."""
    while isinstance(source, (list, tuple)):
        source = source[0] if source else ""
    return os.path.abspath(os.path.expanduser(str(source))) if source else ""


def _overview_source_text(source):
    """Normalize every source path lexically, joined by ``"; "``; one path reads as itself.

    :param source: one path, a list of paths, or paths separated by ``;``.
    """
    return "; ".join(_overview_source_paths(source))


def _overview_geometry_extent(fields):
    """Return known physical well extents without assigning unknown identities."""
    coordinates = tuple((int(field.row), field.column) for field in fields
                        if field.row and field.column > 0)
    return (max((row for row, column in coordinates), default=0),
            max((column for row, column in coordinates), default=0))


def _overview_reference_key(field, mode, conditions):
    """Address a reference without merging identically named wells on plates."""
    if mode == "reference_well":
        return (field.plate, field.well) if field.well else None
    if mode == "reference_row":
        return (field.plate, field.row) if field.row else None
    if mode == "reference_column":
        return (field.plate, field.column) if field.column > 0 else None
    if mode == "reference_condition":
        condition = conditions.get((field.plate, field.well)) if field.well else None
        return (field.plate, condition) if condition else None
    return None


def _overview_normalization_plan(fields, channels, options, read_planes, cancelled=None):
    """Learn display-only bounds from bounded, deterministic raw-pixel samples.

    Sample at most 128 evenly spaced fields and 2048 finite pixels per
    channel and field through the reader's 256-pixel preview limit. Reference
    modes select their scope before sampling. No source arrays are modified.
    """
    from concurrent.futures import CancelledError

    options = dict(options or {})
    mode = options.get("mode", "image")
    if mode not in ("image", "shared", "reference_well", "reference_row",
                    "reference_column", "reference_condition"):
        raise _OverviewValidationError(tr("Choose a supported normalization mode."))
    percentiles = tuple(float(value) for value in options.get("percentiles", (1., 99.)))
    if len(percentiles) != 2 or not 0 <= percentiles[0] < percentiles[1] <= 100:
        raise _OverviewValidationError(tr("The lower percentile must be below the upper percentile."))
    channels = tuple(channels)
    if not channels or len(set(channels)) != len(channels):
        raise _OverviewValidationError(tr("Choose at least one distinct available channel."))
    reference_channel = options.get("reference_channel")
    if reference_channel is not None and reference_channel not in channels:
        raise _OverviewValidationError(tr("The reference channel must be selected for display."))
    mean_match = bool(options.get("mean_match", False))
    if mean_match and reference_channel is None:
        raise _OverviewValidationError(tr("Choose a reference channel before enabling mean-matched display scaling."))
    conditions = dict(options.get("conditions") or {})
    reference = options.get("reference")
    if isinstance(reference, list):
        reference = tuple(reference)
    selected = tuple(fields)
    if mode.startswith("reference_"):
        selected = tuple(field for field in selected
                         if _overview_reference_key(field, mode, conditions) == reference)
        if not selected or reference in (None, ""):
            raise _OverviewValidationError(tr("The selected normalization reference has no fields."))
    indices = np.linspace(0, len(selected) - 1, min(128, len(selected)), dtype=int)
    sampled = tuple(selected[index] for index in indices)
    bounds = {}
    gains = {channel: 1. for channel in channels}
    samples = {channel: [] for channel in channels}
    needs_sample = mode != "image" or reference_channel is not None
    if needs_sample:
        if not sampled:
            raise _OverviewValidationError(tr("No fields are available for normalization."))
        for field in sampled:
            if cancelled is not None and cancelled.is_set():
                raise CancelledError()
            found = set()
            for channel, path, raw in read_planes(field, channels, edge=256):
                if cancelled is not None and cancelled.is_set():
                    raise CancelledError()
                if channel not in samples or channel in found:
                    raise _OverviewValidationError(tr("The normalization sample has ambiguous channels."))
                found.add(channel)
                values = np.asarray(raw).ravel()
                values = values[np.isfinite(values)]
                if values.size:
                    positions = np.linspace(0, values.size - 1, min(2048, values.size), dtype=int)
                    samples[channel].append(np.asarray(values[positions], dtype=np.float64))
            if found != set(channels):
                raise _OverviewValidationError(tr("A selected channel is missing from the normalization sample."))
        pooled = {}
        for channel, pieces in samples.items():
            if not pieces:
                raise _OverviewValidationError(tr("This channel has no finite pixels."))
            pooled[channel] = np.concatenate(pieces)
            if mode != "image":
                bounds[channel] = tuple(float(value) for value in np.percentile(
                    pooled[channel], percentiles))
        if reference_channel is not None and mode != "image":
            reference_bounds = bounds[reference_channel]
            bounds = {channel: reference_bounds for channel in channels}
        if mean_match:
            means = {channel: float(np.mean(values)) for channel, values in pooled.items()}
            if any(not math.isfinite(value) or value <= 0 for value in means.values()):
                raise _OverviewValidationError(tr("Channel-reference scaling requires positive finite sample means."))
            gains = {channel: means[reference_channel] / value for channel, value in means.items()}
    return {"mode": mode, "percentiles": percentiles, "bounds": bounds,
            "gains": gains, "reference": reference, "reference_channel": reference_channel,
            "mean_match": mean_match,
            "sample_fields": len(sampled) if needs_sample else 0,
            "sample_pixels": {channel: sum(len(piece) for piece in pieces)
                              for channel, pieces in samples.items()},
            "max_fields": 128, "max_pixels_per_field": 2048, "preview_edge": 256,
            "field_identities": tuple(field.identity for field in sampled) if needs_sample else ()}


def _overview_normalized_planes(planes, plan, field=None):
    """Return fresh display floats; degenerate bounds/nonfinite pixels become black.

    The transform is clip((gain * x - low) / (high - low), 0, 1).
    Reference-channel bounds are shared; optional mean matching scales raw
    pixels before that affine display transform, never computing pixel ratios.
    """
    if plan["mode"] == "reference_average" and (field is None or
            tuple(plan.get("well_identity", ())) != (field.plate, field.well)):
        raise RuntimeError("Well-mean display rendering requires a resolved per-well plan.")
    planes = tuple(planes)
    image_bounds = {}
    if plan["mode"] == "image":
        for channel, path, raw in planes:
            source = np.asarray(raw)
            values = source[np.isfinite(source)]
            if not values.size:
                raise _OverviewValidationError(tr("This channel has no finite pixels."))
            image_bounds[channel] = np.percentile(values, plan["percentiles"])
        reference_channel = plan.get("reference_channel")
        if reference_channel is not None:
            if reference_channel not in image_bounds:
                raise _OverviewValidationError(tr("The reference channel must be selected for display."))
            image_bounds = {channel: image_bounds[reference_channel] for channel in image_bounds}
    output = []
    for channel, path, raw in planes:
        source = np.asarray(raw)
        finite = np.isfinite(source)
        if not finite.any():
            raise _OverviewValidationError(tr("This channel has no finite pixels."))
        if plan["mode"] == "image":
            low, high = image_bounds[channel]
        else:
            if channel not in plan["bounds"]:
                raise _OverviewValidationError(tr("A selected channel has no learned normalization bounds."))
            low, high = plan["bounds"][channel]
        result = np.zeros(source.shape, dtype=np.float32)
        if high > low:
            values = np.asarray(source[finite], dtype=np.float64)
            values = (plan["gains"].get(channel, 1.) * values - low) / (high - low)
            result[finite] = np.clip(values, 0., 1.).astype(np.float32)
        output.append((channel, path, result))
    return tuple(output)


def _overview_metadata_preview(filenames, pattern):
    """Apply the shared microscope parser to real names without changing files."""
    import re
    from ..regex_detect import apply_regex

    try:
        compiled = re.compile(pattern, re.I)
    except re.error as error:
        raise _OverviewValidationError(str(error)) from error
    names = tuple(Path(name).name for name in filenames)[:20]
    records, missed = apply_regex(names, compiled)
    by_name = {record.filename: record.groups for record in records}
    return tuple((name, dict(by_name.get(name, {}))) for name in names), tuple(missed)


def _overview_matched_fields(fields, pattern, patterns=None):
    """Exclude filename nonmatches while retaining invalid matched well identities.

    :param pattern: the filename regex of every plate without its own.
    :param patterns: ``{plate folder: regex}`` of plates with their own regex.
    """
    import re
    compiled = {}
    kept = []
    for field in fields:
        own = _overview_plate_pattern(patterns, getattr(field, "source", ""), pattern)
        if not own:
            kept.append(field)
            continue
        if own not in compiled:
            compiled[own] = re.compile(own, re.I)
        if all(compiled[own].match(Path(path).name) for _, path in field.files):
            kept.append(field)
    return tuple(kept)


def _overview_acquisition_summary(fields, info, acquisition):
    """Combine acquisition XML metadata with what the filenames and headers show.

    :param fields: matched source records.
    :param info: header information from :func:`_overview_channel_info`.
    :param acquisition: a :func:`spacr.regex_infer._yokogawa_acquisition`
        record, or None.
    :returns: plate geometry, well numbering, channel names and colours,
        pixel size and axis counts, each with where it came from.
    """
    import re
    from ...regex_infer import _channel_display_colour
    acquisition = acquisition or {}
    numeric = any(re.fullmatch(r"\d+", str(value or "")) for value in
                  (_overview_filename_well(path) for field in fields[:50] for _, path in field.files))
    pixel = acquisition.get("pixel_size_um")
    pixel_source = "acquisition" if pixel else ""
    if not pixel:
        pixel = next((entry.get("pixel_size_um") for entry in info.get("files", ())
                      if entry.get("pixel_size_um")), None)
        pixel_source = "image" if pixel else ""
    named = acquisition.get("channels") or {}
    channels = {}
    for position, channel in enumerate(info.get("channels", ())):
        record = dict(named.get(channel) or {})
        name = record.get("name") or ""
        channels[channel] = dict(name=name, excitation_nm=record.get("excitation_nm"),
                                 emission=record.get("emission", ""),
                                 color=record.get("color") or _channel_display_colour(name, position=position))
    rows, columns = acquisition.get("rows"), acquisition.get("columns")
    return dict(
        instrument=acquisition.get("instrument", ""), folder=acquisition.get("folder", ""),
        sources=tuple(acquisition.get("sources", ())), rows=rows, columns=columns,
        numeric_wells=numeric, well_order=acquisition.get("well_order") or ("row" if numeric else None),
        well_table=bool(acquisition.get("wells")), channels=channels,
        pixel_size_um=pixel, pixel_source=pixel_source,
        wells=len({(field.plate, field.well) for field in fields if field.well}),
        fields=len({field.field_id for field in fields if field.field_id}) or acquisition.get("fields"),
        z_planes=len({field.z_id for field in fields if field.z_id}) or max(
            (entry.get("z_count", 1) for entry in info.get("files", ())), default=1),
        time_points=len({field.time_id for field in fields if field.time_id}) or max(
            (entry.get("time_count", 1) for entry in info.get("files", ())), default=1),
        acquired_z=acquisition.get("z_slices"), z_step_um=acquisition.get("z_step_um"),
        acquired_time=acquisition.get("time_points"))


def _overview_filename_well(path):
    """Return the CQ1 well token a filename carries, or ``""``."""
    import re
    from ...regex_infer import _CQ1_FILENAME
    match = re.match(_CQ1_FILENAME, Path(path).name, re.I)
    return match.group("wellID") if match else ""


def _overview_wizard_probe(source, pattern, convention, cancelled, source_scope="folder", well_layout=None,
                           use_subfolders=(), excluded=(), patterns=None):
    """Find the plates, index names on a worker and inspect only the first field's image headers.

    One source path whose saved view state lists more plate folders brings
    those back too, so reopening an Overview restores every plate it had.
    """
    import re
    from ...regex_infer import _CQ1_FILENAME, _yokogawa_acquisition
    paths = _overview_source_paths(source)
    restored = []
    patterns = dict(patterns or {})
    if len(paths) == 1:
        state = _overview_load_view_state(paths[0])
        saved = state.get("sources") or ()
        restored = [path for path in _overview_source_paths(saved) if path not in paths and os.path.isdir(path)]
        for folder, value in dict((state.get("options") or {}).get("plate_regex") or {}).items():
            patterns.setdefault(str(folder), str(value))
    sources = paths + restored
    fields, detection = _overview_index_sources(sources, pattern, convention, cancelled, source_scope=source_scope,
                                                well_layout=well_layout, use_subfolders=use_subfolders,
                                                excluded=excluded, patterns=patterns)
    if cancelled.is_set():
        return fields, None
    if not fields:
        return fields, {"channels": (), "plates": detection["plates"], "plate_messages": detection["messages"],
                        "sources": sources, "restored_sources": restored, "plate_regex": patterns}
    detected_pattern, detected_label = pattern, convention
    all_names = tuple(Path(path).name for field in fields for _, path in field.files)
    if (convention == "auto" and not pattern
            and 2 * sum(bool(re.match(_CQ1_FILENAME, name, re.I)) for name in all_names) >= len(all_names)):
        detected_pattern, detected_label = _CQ1_FILENAME, "cq1"
    elif convention == "auto" and not pattern:
        from ..regex_detect import auto_detect_regex
        names = all_names[:20]
        detected_pattern, detected_label, hits = auto_detect_regex(names)
        if detected_pattern and ("wellID" not in re.compile(detected_pattern).groupindex
                or not any(re.match(detected_pattern, name, re.I) for name in names)):
            detected_pattern, detected_label = pattern, convention
        if detected_pattern and hits:
            inferred = _overview_index_sources(sources, detected_pattern, "auto", cancelled,
                                               source_scope=source_scope, well_layout=well_layout,
                                               use_subfolders=use_subfolders, excluded=excluded)[0]
            by_path = {path: (field, channel) for field in inferred for channel, path in field.files}
            preserves_metadata = all(
                path in by_path
                and all(not getattr(field, axis) or getattr(field, axis) == getattr(by_path[path][0], axis)
                        for axis in ("well", "field_id", "time_id", "z_id"))
                and (not channel or channel == by_path[path][1])
                for field in fields for channel, path in field.files)
            preserves_groups = preserves_metadata and all(
                {path for _, path in by_path[field.files[0][1]][0].files}
                == {path for _, path in field.files} for field in fields)
            if preserves_metadata and preserves_groups:
                fields = inferred
            else:
                detected_pattern, detected_label = pattern, convention
    if cancelled.is_set() or not fields:
        return fields, None
    matched = _overview_matched_fields(fields, detected_pattern, patterns)
    first = next((field for field in matched if field.well and not field.unresolved),
                 next(iter(matched), fields[0]))
    info = dict(_overview_channel_info(first))
    info["detected_pattern"] = detected_pattern or ""
    info["detected_label"] = detected_label
    acquisition = _yokogawa_acquisition(Path(first.files[0][1]).parent) if first.files else None
    info["acquisition"] = _overview_acquisition_summary(matched or fields, info, acquisition)
    info["plates"] = detection["plates"]
    info["plate_messages"] = detection["messages"]
    info["sources"] = sources
    info["restored_sources"] = restored
    info["plate_regex"] = patterns
    return fields, info


def _overview_wizard_scan(source, pattern, convention, cancelled, source_scope="folder", well_layout=None,
                          use_subfolders=(), excluded=(), patterns=None):
    """Probe the wizard's source and return a refused source as a message, not an exception.

    A source that is not yet a folder (while a path is typed, or after it was
    moved) or a pattern without a wellID group is a settings answer for the
    wizard's status line. Raised on the worker, it was filed as an error
    report (GitHub #167).

    :returns: ``(fields, info, refusal)``; ``refusal`` is ``""`` unless the
        source or pattern was refused, when ``fields`` is empty and ``info``
        is ``None``.
    """
    try:
        fields, info = _overview_wizard_probe(source, pattern, convention, cancelled, source_scope, well_layout,
                                              use_subfolders, excluded, patterns)
    except _OverviewValidationError as error:
        return (), None, str(error)
    return fields, info, ""


class _OverviewMetadataWizard(QWidget):
    """Choose a real source's metadata, axes, geometry and display transform."""

    accepted = Signal()
    rejected = Signal()
    finished = Signal(int)

    def __init__(self, source="", settings=None, parent=None):
        """Build the source wizard; filesystem work starts only on its worker."""
        import threading
        from PySide6.QtWidgets import (QFormLayout, QLineEdit, QSpinBox,
                                       QDoubleSpinBox, QTableWidget,
                                       QHeaderView, QScrollArea)
        from ..regex_detect import BUILTIN_REGEXES
        from .sortable_table import install_sorting
        from .card import Card
        from PySide6.QtWidgets import QSplitter

        super().__init__(parent)
        from ..gil_priority import hold_background_imports

        hold_background_imports(self, lambda wizard: not wizard._finished and wizard.isVisible())
        self.setWindowTitle(tr("Image metadata and display"))
        self._settings = dict(settings or {})
        self._fields = ()
        self._info = None
        self._cancelled = threading.Event()
        self._generation = 0
        self._finished = False
        self._result = QDialog.Rejected
        self._runner = JobRunner(self, threaded=True, app_key="overview metadata", user_visible=False)
        self._runner.job_failed.connect(self._failed)
        self._readout_timer = QTimer(self)
        self._readout_timer.setSingleShot(True)
        self._readout_timer.timeout.connect(self._fit_readouts)
        shell = QVBoxLayout(self)
        body = QWidget()
        self._settings_content = body
        outer = QVBoxLayout(body)
        scroll = QScrollArea()
        self._settings_scroll = scroll
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        split = QSplitter()
        split.setChildrenCollapsible(False)
        split.addWidget(scroll)
        preview_body = QWidget()
        preview_column = QVBoxLayout(preview_body)
        split.addWidget(preview_body)
        split.setStretchFactor(1, 2)
        self._settings_dragged = False
        split.splitterMoved.connect(self._settings_moved)
        shell.addWidget(split, 1)
        self.resize(920, 760)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.WrapAllRows)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.source = QLineEdit(str(source or ""))
        self.source.setObjectName("OverviewWizardSource")
        source_row = QHBoxLayout()
        source_row.addWidget(self.source, 1)
        browse = QPushButton(tr("Choose folder…"))
        browse.clicked.connect(self._browse)
        source_row.addWidget(browse)
        add_plates = QPushButton(tr("Add folders…"))
        add_plates.setObjectName("OverviewWizardAddFolders")
        add_plates.setToolTip(tr("Add more plate folders, or folders of plates; several can be selected at once."))
        add_plates.clicked.connect(self._choose_more_sources)
        source_row.addWidget(add_plates)
        form.addRow(tr("Image source"), source_row)
        self._use_subfolders = set(self._settings.get("use_subfolders") or ())
        self._excluded_plates = set(self._settings.get("excluded_plates") or ())
        self._plate_names = dict(self._settings.get("plate_names") or {})
        self._plate_regex = {str(key): str(value) for key, value in
                             dict(self._settings.get("plate_regex") or {}).items() if value}
        self._detected_plates = []
        self.plate_table = QTableWidget(0, 5)
        self.plate_table.setObjectName("OverviewWizardPlates")
        install_sorting(self.plate_table)
        self.plate_table.setHorizontalHeaderLabels([tr("Plate"), tr("Images"), tr("Folder"), tr("Use subfolders"),
                                                    tr("Filename regex")])
        self.plate_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.plate_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.plate_table.setTextElideMode(Qt.ElideMiddle)
        self.plate_table.setMinimumHeight(110)
        self.plate_table.setToolTip(tr(
            "The plates found in the source paths: a folder with images in it is a plate, otherwise each "
            "subfolder with images is one. Double-click a name to rename the plate."))
        self.plate_table.itemChanged.connect(self._plate_renamed)
        self.plate_summary = QLabel("")
        self.plate_summary.setObjectName("OverviewWizardPlateSummary")
        self.plate_summary.setWordWrap(True)
        remove_plates = QPushButton(tr("Remove selected plates"))
        remove_plates.setObjectName("OverviewWizardRemovePlates")
        remove_plates.clicked.connect(self._remove_selected_plates)
        form.addRow(tr("Plates"), self.plate_table)
        form.addRow(self.plate_summary)
        form.addRow(remove_plates)
        self.scope = QComboBox()
        self.scope.setObjectName("OverviewWizardSourceScope")
        self.scope.addItem(tr("This folder only (raw images at this level)"), "folder")
        self.scope.addItem(tr("Include subfolders (may include derived images)"), "recursive")
        self.scope.currentIndexChanged.connect(self._invalidate)
        form.addRow(tr("Source scope"), self.scope)
        self.preset = QComboBox()
        self.preset.setObjectName("OverviewWizardMicroscope")
        self.preset.addItem(tr("Automatic filename metadata"), "auto")
        for key in BUILTIN_REGEXES:
            self.preset.addItem(tr(key), key)
        self.preset.addItem(tr("Custom filename regex"), "custom")
        self.regex = QLineEdit(str(self._settings.get("custom_regex") or ""))
        self.regex.setObjectName("OverviewWizardRegex")
        self.regex.setProperty("spacrNoFieldFade", True)
        form.addRow(tr("Microscope preset"), self.preset)
        form.addRow(tr("Filename regex"), self.regex)
        self.preset.currentIndexChanged.connect(self._preset_changed)
        preview = QPushButton(tr("Preview metadata"))
        preview.clicked.connect(self._scan)
        form.addRow(preview)
        source_card = Card(title=tr("Source"), foldable=True)
        source_card.body_layout.addLayout(form)
        outer.addWidget(source_card)
        self.preview = QTableWidget(0, 9)
        self.preview.setObjectName("OverviewWizardParsePreview")
        install_sorting(self.preview)
        self.preview.setHorizontalHeaderLabels([tr("File"), tr("Plate"), tr("Well"), tr("Row"), tr("Column"),
                                                tr("Field"), tr("Time"), tr("Z"), tr("Channel")])
        self.preview.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        for column, width in ((0, 180), (1, 100)):
            self.preview.horizontalHeader().setSectionResizeMode(column, QHeaderView.Interactive)
            self.preview.setColumnWidth(column, width)
        self.preview.setMinimumHeight(240)
        self.preview.setEditTriggers(QTableWidget.NoEditTriggers)
        preview_column.addWidget(self.preview, 1)
        self.axes = QLabel("")
        self.axes.setWordWrap(True)
        preview_column.addWidget(self.axes)
        layout = QFormLayout()
        layout.setRowWrapPolicy(QFormLayout.WrapAllRows)
        layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.channels = QLineEdit()
        self.channels.setObjectName("OverviewWizardChannels")
        self.channels.setToolTip(tr("Choose only channel IDs reported by this image's metadata."))
        layout.addRow(tr("Display channels"), self.channels)
        self.channel_table = QTableWidget(0, 4)
        self.channel_table.setObjectName("OverviewWizardChannelTable")
        install_sorting(self.channel_table)
        self.channel_table.setHorizontalHeaderLabels([tr("Channel"), tr("Name"), tr("Wavelength"), tr("Colour")])
        self.channel_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.channel_table.horizontalHeader().setStretchLastSection(True)
        self.channel_table.setMinimumHeight(110)
        self.channel_table.setToolTip(tr(
            "Channel names and display colours, filled from the acquisition metadata or the wavelengths. "
            "Edit a name or a colour (#rrggbb or a colour name) to override it. In the CMY and CMYK "
            "colour modes, cyan, magenta, yellow or black picks the ink a channel prints with."))
        self.channel_table.itemChanged.connect(self._channel_colour_edited)
        layout.addRow(tr("Channel names and colours"), self.channel_table)
        self.pixel_size = QDoubleSpinBox()
        self.pixel_size.setObjectName("OverviewWizardPixelSize")
        self.pixel_size.setRange(0., 1000.)
        self.pixel_size.setDecimals(4)
        self.pixel_size.setSingleStep(.01)
        self.pixel_size.setSuffix(" µm")
        self.pixel_size.setSpecialValueText(tr("Unknown"))
        self.pixel_size.setToolTip(tr(
            "Pixel size in micrometres, filled from the acquisition metadata or the image header when they state it."))
        self._pixel_size_explicit = self._settings.get("pixel_size_um") is not None
        if self._pixel_size_explicit:
            self.pixel_size.setValue(float(self._settings["pixel_size_um"]))
        self.pixel_size.valueChanged.connect(self._pixel_size_edited)
        layout.addRow(tr("Pixel size"), self.pixel_size)
        self.time = QSpinBox()
        self.time.setObjectName("OverviewWizardTime")
        self.z_mode = QComboBox()
        self.z_mode.addItem(tr("Single Z plane"), "plane")
        self.z_mode.addItem(tr("Maximum Z projection"), "max")
        self.z_mode.addItem(tr("Mean Z projection"), "mean")
        self.z = QSpinBox()
        self.z.setObjectName("OverviewWizardZ")
        layout.addRow(tr("Time index"), self.time)
        layout.addRow(tr("Z view"), self.z_mode)
        layout.addRow(tr("Z index"), self.z)
        self.plate_geometry = QComboBox()
        self.plate_geometry.setObjectName("OverviewWizardGeometry")
        self.plate_geometry.addItem(tr("Coverslip"), (1, 1))
        for count, dimensions in ((6, (2, 3)), (12, (3, 4)), (24, (4, 6)),
                                  (48, (6, 8)), (96, (8, 12)), (384, (16, 24)),
                                  (1536, (32, 48))):
            self.plate_geometry.addItem(tr("{count}-well plate", count=count), dimensions)
        self._geometry_explicit = self._settings.get("geometry") is not None
        selected_geometry = self._settings.get("geometry")
        selected_index = next((index for index in range(self.plate_geometry.count())
                               if selected_geometry and
                               self.plate_geometry.itemData(index) == tuple(selected_geometry)), -1)
        self.plate_geometry.setCurrentIndex(selected_index if selected_index >= 0 else 5)
        self.plate_geometry.activated.connect(self._choose_geometry)
        layout.addRow(tr("Plate geometry"), self.plate_geometry)
        self.well_order = QComboBox()
        self.well_order.setObjectName("OverviewWizardWellOrder")
        self.well_order.addItem(tr("Row by row (A01, A02, …)"), "row")
        self.well_order.addItem(tr("Column by column (A01, B01, …)"), "column")
        self.well_order.setToolTip(tr(
            "How numeric well indices such as CQ1 W0049 are counted across the plate geometry. "
            "Filled from the acquisition metadata; changing it or the geometry places the wells again."))
        self.well_order.setEnabled(False)
        self._well_layout = self._settings.get("well_layout")
        self._layout_explicit = bool(self._well_layout)
        if self._layout_explicit and len(self._well_layout) > 2:
            self.well_order.setCurrentIndex(max(0, self.well_order.findData(self._well_layout[2])))
        self.well_order.activated.connect(self._choose_well_order)
        layout.addRow(tr("Well index order"), self.well_order)
        self.representatives = QSpinBox()
        self.representatives.setRange(1, 100)
        self.representatives.setValue(1)
        layout.addRow(tr("Representative fields per well"), self.representatives)
        self.mode = QComboBox()
        for label, value in ((tr("Percentiles per image"), "image"),
                             (tr("Shared channel percentile bounds"), "shared"),
                             (tr("Bounds from a reference well"), "reference_well"),
                             (tr("Bounds from a reference row"), "reference_row"),
                             (tr("Bounds from a reference column"), "reference_column"),
                             (tr("Bounds from a reference condition"), "reference_condition"),
                             (tr("Well-mean display scaling to reference average"), "reference_average")):
            self.mode.addItem(label, value)
        self.mode.currentIndexChanged.connect(self._reference_choices)
        layout.addRow(tr("Display normalization"), self.mode)
        self.low = QDoubleSpinBox()
        self.high = QDoubleSpinBox()
        percentiles = (self._settings.get("normalization") or {}).get("percentiles")
        percentiles = percentiles or self._settings.get("normalization_percentiles") or (1., 99.9)
        percentiles = _overview_normalization_plan((), (0,),
            {"mode": "image", "percentiles": percentiles}, None)["percentiles"]
        for spin, value in ((self.low, percentiles[0]), (self.high, percentiles[1])):
            spin.setRange(0., 100.)
            spin.setDecimals(3)
            spin.setValue(value)
        layout.addRow(tr("Lower normalization percentile"), self.low)
        layout.addRow(tr("Upper normalization percentile"), self.high)
        self.reference_kind = QComboBox()
        for label, value in ((tr("Well"), "reference_well"), (tr("Row"), "reference_row"),
                             (tr("Column"), "reference_column"), (tr("Condition"), "reference_condition")):
            self.reference_kind.addItem(label, value)
        self.reference_kind.setCurrentIndex(2)
        self.reference_kind.setEnabled(False)
        self.reference_kind.currentIndexChanged.connect(self._reference_choices)
        layout.addRow(tr("Reference group"), self.reference_kind)
        self.reference = QComboBox()
        layout.addRow(tr("Reference scope"), self.reference)
        self.gain = QComboBox()
        self.gain.addItem(tr("Use each channel’s own bounds"), None)
        layout.addRow(tr("Use reference-channel bounds"), self.gain)
        self.mean_match = Toggle(tr("Mean-matched display scaling before reference bounds"))
        layout.addRow(self.mean_match)
        display_card = Card(title=tr("Display"), foldable=True)
        display_card.body_layout.addLayout(layout)
        outer.addWidget(display_card)
        self.conditions = QTableWidget(0, 3)
        self.conditions.setObjectName("OverviewWizardConditions")
        install_sorting(self.conditions)
        self.conditions.setHorizontalHeaderLabels([tr("Plate"), tr("Well"), tr("Condition")])
        self.conditions.itemChanged.connect(self._reference_choices)
        annotations_card = Card(title=tr("Annotations"), foldable=True)
        annotations_card.body_layout.addWidget(self.conditions)
        outer.addWidget(annotations_card)
        explanation = QLabel(tr(
            "Display only: clip((gain × pixel − lower bound) / (upper bound − lower bound), 0, 1). "
            "Equal lower/upper bounds and nonfinite pixels are black; channels without finite pixels are refused. "
            "Shared/reference bounds use at most 128 evenly spaced fields and 2048 finite "
            "sample pixels per channel and field, from bounded 256-pixel previews. "
            "Optional mean-matched display scaling uses the reference-channel raw sample mean divided by each channel raw sample mean, including zeros. The same reference-channel bounds are then used; "
            "both must be positive and finite. It is not a per-pixel ratio. Raw values stay unchanged."))
        explanation.setWordWrap(True)
        display_card.body_layout.addWidget(explanation)
        well_explanation = QLabel(tr(
            "Well-mean scaling uses the selected representative fields and time/Z view. "
            "Finite original pixels, including zeros, are averaged equally by field, then equally by reference well on the same plate. "
            "Each well is scaled to that reference mean before fixed reference bounds and clipping; clipped display means need not match. "
            "Bounds use balanced original-pixel samples from reference wells. Nonreference well statistics are learned on demand. Raw CSV values stay unchanged."))
        well_explanation.setWordWrap(True)
        display_card.body_layout.addWidget(well_explanation)
        outer.addStretch(1)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        preview_column.addWidget(self.status)
        self._module_actions = QWidget()
        self._action_bar = self._module_actions
        actions = QHBoxLayout(self._module_actions)
        cancel = QPushButton(tr("Cancel"))
        cancel.clicked.connect(self.reject)
        self.next = QPushButton(tr("Next"))
        self.next.setObjectName("OverviewWizardNext")
        self.next.setEnabled(False)
        self.next.clicked.connect(self._next)
        actions.addStretch(1)
        actions.addWidget(cancel)
        actions.addWidget(self.next)
        shell.addWidget(self._module_actions)
        self._wizard_shell, self._wizard_body = shell, outer
        self._wizard_scroll = scroll
        self._wizard_content = split
        self._action_buttons = (cancel, self.next)
        self._compact_scroll = QScrollArea(self)
        self._compact_scroll.setWidgetResizable(True)
        self._compact_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._compact_page = QWidget()
        self._compact_layout = QVBoxLayout(self._compact_page)
        self._compact_layout.setContentsMargins(0, 0, 0, 0)
        self._compact_scroll.setWidget(self._compact_page)
        self._compact_scroll.hide()
        self._sticky_margins = shell.contentsMargins()
        self._actions_scroll = False
        self._fitting_actions = False
        self._action_fit_timer = QTimer(self)
        self._action_fit_timer.setSingleShot(True)
        self._action_fit_timer.timeout.connect(self._fit_actions)
        self._host_actions = QWidget(self)
        host_actions = QHBoxLayout(self._host_actions)
        for caption, callback in ((tr("Preview metadata"), self._scan), (tr("Cancel"), self.reject)):
            button = QPushButton(caption)
            button.clicked.connect(callback)
            host_actions.addWidget(button)
        self._host_actions.hide()
        from PySide6.QtWidgets import QSizePolicy
        for combo in body.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
            combo.setMinimumWidth(0)
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.source.textEdited.connect(self._invalidate)
        self.regex.textEdited.connect(self._custom_regex_changed)
        if self.regex.text():
            self.preset.setCurrentIndex(self.preset.count() - 1)
        from ..screens.settings_model import retarget_field_tooltips
        retarget_field_tooltips(self)
        self._startup_timer = QTimer(self)
        self._startup_timer.setSingleShot(True)
        self._startup_timer.timeout.connect(self._scan)
        self._rescan_timer = QTimer(self)
        self._rescan_timer.setSingleShot(True)
        self._rescan_timer.timeout.connect(self._scan)
        self._startup_timer.start(0)
        self._action_fit_timer.start(0)

    def event(self, event):
        """Coalesce actual allocation and native font/style changes after Qt handles them."""
        result = super().event(event)
        timer = getattr(self, "_action_fit_timer", None)
        if (timer is not None and not self._finished and not self._fitting_actions
                and event.type() in (QEvent.Resize, QEvent.FontChange,
                                     QEvent.StyleChange, QEvent.LayoutRequest)):
            timer.start(0)
        return result

    def _fit_actions(self):
        """Scroll the same actions in short allocations and restore sticky actions on growth.

        Fresh native footer hints reserve a full footer-height content viewport.
        Only compact mode drops redundant outer margins. Its vertical page
        scroll keeps the footer separate from the form's horizontal scrollbar.
        """
        if self._finished or self._fitting_actions:
            return
        self._widen_settings()
        self._fitting_actions = True
        try:
            self._fit_action_buttons()
            self._action_bar.layout().invalidate()
            from ..gui_scale import _set_actual_dim
            action_margins = self._action_bar.layout().contentsMargins()
            action_height = max(button.property("minimumSize").height()
                                for button in self._action_buttons)
            _set_actual_dim(self._action_bar, "minh", action_height
                            + action_margins.top() + action_margins.bottom())
            footer_height = max(self._action_bar.sizeHint().height(),
                                self._action_bar.minimumSizeHint().height())
            margins = self._sticky_margins
            from ..gui_scale import _original_for
            margin_height = margins.top() + margins.bottom()
            needed = 2 * footer_height + margin_height + max(0, self._wizard_shell.spacing())
            compact = QWidget.height(self) < needed
            if compact == self._actions_scroll:
                return
            set_margins = (_original_for(self._wizard_shell, "setContentsMargins")
                           or type(self._wizard_shell).setContentsMargins)
            if compact:
                self._wizard_shell.removeWidget(self._action_bar)
                self._wizard_shell.removeWidget(self._wizard_content)
                self._compact_layout.addWidget(self._wizard_content, 1)
                self._compact_layout.addWidget(self._action_bar)
                self._wizard_shell.addWidget(self._compact_scroll, 1)
                self._compact_scroll.show()
                set_margins(self._wizard_shell, 0, 0, 0, 0)
            else:
                self._compact_layout.removeWidget(self._action_bar)
                self._compact_layout.removeWidget(self._wizard_content)
                self._wizard_shell.removeWidget(self._compact_scroll)
                self._wizard_shell.addWidget(self._wizard_content, 1)
                self._wizard_shell.addWidget(self._action_bar)
                self._compact_scroll.hide()
                set_margins(self._wizard_shell, margins)
            self._actions_scroll = compact
            self._wizard_content.show()
            self._action_bar.show()
        finally:
            self._fitting_actions = False

    def _fit_action_buttons(self):
        """Reserve actual native text slots from fresh hints, permitting later shrink.

        Only minimum dimensions change; current font, style, default-button
        features and icon insets are measured without scaling them a second time.
        """
        _fit_overview_action_buttons(self._action_buttons)

    def _invalidate(self, *_):
        """Debounce filename previews and prevent acceptance of stale metadata."""
        self._generation += 1
        self._cancelled.set()
        self.next.setEnabled(False)
        if hasattr(self, "_rescan_timer") and not self._finished:
            self._rescan_timer.start(300)

    def _browse(self):
        """Choose another folder while preserving the old source on cancellation."""
        from PySide6.QtWidgets import QFileDialog
        path = QFileDialog.getExistingDirectory(self, tr("Image source"))
        if path:
            self.source.setText(path)
            self._scan()

    def _choose_more_sources(self, *_):
        """Pick one or more further folders and add them to the source.

        :returns: the chosen paths.
        """
        dialog = _overview_folders_dialog(self, tr("Add folders…"))
        self._add_folders_dialog = dialog
        paths = dialog.selectedFiles() if dialog.exec() else []
        if paths:
            self._add_source_paths(paths)
        return paths

    def _add_source_paths(self, paths):
        """Add folders to the source and preview the plates again."""
        self.source.setText("; ".join(_overview_source_paths([self.source.text(), *paths])))
        self._scan()

    def _show_plates(self, info):
        """List the detected plates with their image counts, before Next."""
        from .sortable_table import table_item

        plates = list((info or {}).get("plates") or ())
        restored = list((info or {}).get("restored_sources") or ())
        if restored:
            self.source.blockSignals(True)
            self.source.setText("; ".join((info or {}).get("sources") or ()))
            self.source.blockSignals(False)
        for folder, value in dict((info or {}).get("plate_regex") or {}).items():
            if value:
                self._plate_regex.setdefault(str(folder), str(value))
        self._detected_plates = plates
        self.plate_table.blockSignals(True)
        self.plate_table.setRowCount(0)
        self.plate_table.setRowCount(len(plates))
        for row, plate in enumerate(plates):
            name = table_item(str(self._plate_names.get(plate["default_name"]) or plate["default_name"]))
            name.setData(Qt.UserRole, plate["default_name"])
            name.setToolTip(plate["path"])
            self.plate_table.setItem(row, 0, name)
            for column, value in ((1, str(plate["images"]) if plate["images"] else tr("in subfolders")),
                                  (2, plate["path"])):
                item = table_item(value)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setToolTip(plate["path"])
                self.plate_table.setItem(row, column, item)
            if plate["mixed"] or plate["path"] in self._use_subfolders:
                toggle = Toggle()
                toggle.setChecked(plate["path"] in self._use_subfolders)
                toggle.setToolTip(tr("This folder holds images and also subfolders of images. Switch on to "
                                     "load each subfolder as a plate instead."))
                toggle.toggled.connect(functools.partial(self._use_plate_subfolders, plate["path"]))
                self.plate_table.setCellWidget(row, 3, toggle)
            else:
                self.plate_table.removeCellWidget(row, 3)
            own = _overview_plate_pattern(self._plate_regex, plate["path"], "")
            regex = table_item(own)
            regex.setData(Qt.UserRole, plate["path"])
            regex.setToolTip(tr("The filename regex this plate is read with; empty uses the filename regex above, "
                                "the first plate's."))
            self.plate_table.setItem(row, 4, regex)
        self.plate_table.blockSignals(False)
        notes = [tr("{count} plates found.", count=len(plates)) if len(plates) != 1 else tr("One plate found.")]
        mixed = [plate["path"] for plate in plates if plate["mixed"]]
        if mixed:
            notes.append(tr("{paths} holds images and subfolders of images; it is loaded as one plate unless "
                            "Use subfolders is switched on.", paths=", ".join(mixed)))
        if restored:
            notes.append(tr("Restored {count} more source paths saved with this Overview.", count=len(restored)))
        notes.extend((info or {}).get("plate_messages") or ())
        self.plate_summary.setText(" ".join(notes))

    def _plate_renamed(self, item):
        """Remember a plate's new display name, or its own filename regex."""
        if item.column() == 4:
            self._plate_regex_changed(item)
            return
        if item.column() != 0:
            return
        default = item.data(Qt.UserRole)
        text = item.text().strip()
        if default:
            if text and text != default:
                self._plate_names[default] = text
            else:
                self._plate_names.pop(default, None)

    def _plate_regex_changed(self, item):
        """Read one plate with its own filename regex, or with the first plate's again, and preview again."""
        import re

        path = str(item.data(Qt.UserRole) or "")
        text = item.text().strip()
        for key in [key for key in self._plate_regex if os.path.realpath(key) == os.path.realpath(path)]:
            self._plate_regex.pop(key)
        if text and text != self.regex.text().strip():
            try:
                re.compile(text)
            except re.error as error:
                self.status.setText(tr("Invalid filename regex: {error}", error=error))
                return
            self._plate_regex[path] = text
        self._scan()

    def _use_plate_subfolders(self, path, on):
        """Load a mixed folder's subfolders as plates, or the folder itself, and preview again."""
        if on:
            self._use_subfolders.add(path)
        else:
            self._use_subfolders.discard(path)
        self._scan()

    def _remove_selected_plates(self, *_):
        """Drop the selected plates from the source and preview again."""
        rows = sorted({index.row() for index in self.plate_table.selectedIndexes()})
        chosen = {self.plate_table.item(row, 0).data(Qt.UserRole) for row in rows
                  if self.plate_table.item(row, 0) is not None}
        folders = [plate["path"] for plate in self._detected_plates if plate["default_name"] in chosen]
        if not folders:
            return
        given = _overview_source_paths(self.source.text())
        self._excluded_plates.update(path for path in folders if path not in given)
        kept = [path for path in given if path not in folders]
        self.source.setText("; ".join(kept))
        if kept:
            self._scan()

    def _preset_changed(self):
        """Use the exact filename patterns offered by the Generate Masks editor."""
        from ..regex_detect import BUILTIN_REGEXES
        key = self.preset.currentData()
        if key != "custom":
            self.regex.setText(BUILTIN_REGEXES.get(key, ""))
        self._invalidate()

    def _custom_regex_changed(self):
        """Mark a typed pattern as custom without replacing its contents."""
        self.preset.blockSignals(True)
        self.preset.setCurrentIndex(self.preset.count() - 1)
        self.preset.blockSignals(False)
        self._invalidate()

    def _scan(self):
        """Queue cancellable indexing and header inspection without decoding images."""
        import threading
        if self._finished:
            return
        self._invalidate()
        self._rescan_timer.stop()
        self._cancelled = threading.Event()
        generation = self._generation
        source, pattern = self.source.text(), self.regex.text()
        source_scope = self.scope.currentData()
        convention = self.preset.currentData()
        cancelled = self._cancelled
        self.status.setText(tr("Reading filename metadata and image headers…"))
        well_layout = self._well_layout if self._layout_explicit else None
        use_subfolders, excluded = tuple(self._use_subfolders), tuple(self._excluded_plates)
        patterns = dict(self._plate_regex)
        self._runner.submit(lambda: (generation, _overview_wizard_scan(
            source, pattern, "auto" if convention == "custom" else convention, cancelled, source_scope,
            well_layout, use_subfolders, excluded, patterns)), self._scanned)

    def _scanned(self, payload):
        """Expose actual parsed identities, available channels and embedded axes."""
        from .sortable_table import table_item
        from PySide6.QtGui import QColor
        from ..theme import active_palette
        generation, (fields, info, refusal) = payload
        if generation != self._generation or self._cancelled.is_set():
            return
        self._fields, self._info = fields, info
        self._show_plates(info)
        if refusal:
            self.status.setText(refusal)
            self.next.setEnabled(False)
            return
        if not fields or not info or not info["channels"]:
            self._info = None
            self.status.setText(tr("No readable image fields or available channels were found."))
            return
        detected = info.get("detected_pattern", "")
        if detected and not self.regex.text():
            self.regex.blockSignals(True)
            self.regex.setText(detected)
            self.regex.blockSignals(False)
            index = self.preset.findData(info.get("detected_label"))
            self.preset.blockSignals(True)
            self.preset.setCurrentIndex(index if index >= 0 else self.preset.count() - 1)
            self.preset.blockSignals(False)
        self._fields = _overview_matched_fields(fields, self.regex.text(), self._plate_regex)
        all_files = tuple(dict.fromkeys(path for field in fields for _, path in field.files))
        files = all_files[:20]
        pattern = self.regex.text()
        if pattern:
            rows, missed = _overview_metadata_preview(files, pattern)
        else:
            rows = tuple((Path(path).name, {}) for path in files)
            missed = ()
        self.preview.setRowCount(len(rows))
        self.preview.setSortingEnabled(False)
        placed = {path: field for field in self._fields for _, path in field.files}
        for row, (name, values) in enumerate(rows):
            from ...schema import parse_well, letters_from_row_index
            well = values.get("wellID", "")
            if str(well).isdigit() and files[row] in placed and placed[files[row]].well:
                well = placed[files[row]].well
            try:
                physical_row, physical_column = parse_well(well, strict=True)
                row_letters = letters_from_row_index(int(physical_row[1:]))
                column_number = str(int(physical_column[1:]))
            except (ValueError, TypeError):
                row_letters, column_number = "", ""
            for column, value in enumerate((name, values.get("plateID", ""), well, row_letters, column_number)
                    + tuple(values.get(key, "") or "" for key in ("fieldID", "timeID", "sliceID", "chanID"))):
                item = table_item(str(value))
                item.setToolTip(str(value))
                if name in missed:
                    item.setForeground(QColor(active_palette()["fg_dim"]))
                self.preview.setItem(row, column, item)
        self.preview.setSortingEnabled(True)
        available = tuple(info["channels"])
        requested = self._settings.get("channels") or ()
        chosen = tuple(channel for channel in requested if channel in available) or available[:3]
        self.channels.setText(", ".join(str(channel) for channel in chosen))
        self.gain.clear()
        self.gain.addItem(tr("Use each channel’s own bounds"), None)
        for channel in available:
            self.gain.addItem(str(channel), channel)
        descriptions = []
        for entry in info["files"]:
            descriptions.append(tr("Axes: {axes}; shape: {shape}; type: {dtype}", axes=entry["axes"],
                                   shape=str(entry["shape"]), dtype=entry["dtype"]))
        acquisition = info.get("acquisition") or {}
        self.axes.setText(" ".join((tr("First field: {details}. Available channel IDs: {channels}",
                                       details="; ".join(dict.fromkeys(descriptions)), channels=str(available)),)
                                   + self._acquisition_sentences(acquisition)))
        self._fill_channel_table(available, acquisition.get("channels") or {})
        if not self._pixel_size_explicit:
            self.pixel_size.blockSignals(True)
            self.pixel_size.setValue(float(acquisition.get("pixel_size_um") or 0.))
            self.pixel_size.blockSignals(False)
        numeric = bool(acquisition.get("numeric_wells"))
        self.well_order.setEnabled(numeric)
        if numeric and not self._layout_explicit:
            self.well_order.setCurrentIndex(max(0, self.well_order.findData(
                acquisition.get("well_order") if acquisition.get("well_order") in ("row", "column") else "row")))
        self.time.setRange(0, max(entry.get("time_count", 1) for entry in info["files"]) - 1)
        self.z.setRange(0, max(entry.get("z_count", 1) for entry in info["files"]) - 1)
        if not self._geometry_explicit and acquisition.get("rows") and acquisition.get("columns"):
            dimensions = (int(acquisition["rows"]), int(acquisition["columns"]))
            index = next((index for index in range(self.plate_geometry.count())
                          if self.plate_geometry.itemData(index) == dimensions), -1)
            if index < 0:
                self.plate_geometry.addItem(tr("{rows} × {columns} plate", rows=dimensions[0],
                                               columns=dimensions[1]), dimensions)
                index = self.plate_geometry.count() - 1
            self.plate_geometry.setCurrentIndex(index)
        elif not self._geometry_explicit:
            rows_required, columns_required = _overview_geometry_extent(self._fields)
            for index in range(self.plate_geometry.count()):
                rows_available, columns_available = self.plate_geometry.itemData(index)
                if rows_available >= rows_required and columns_available >= columns_required:
                    self.plate_geometry.setCurrentIndex(index)
                    break
        wells = sorted({(field.plate, field.well) for field in self._fields if field.well})
        self.conditions.blockSignals(True)
        self.conditions.setSortingEnabled(False)
        self.conditions.setRowCount(len(wells))
        for row, pair in enumerate(wells):
            for column, value in enumerate(pair + ("",)):
                item = table_item(str(value))
                if column < 2:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.conditions.setItem(row, column, item)
        self.conditions.setSortingEnabled(True)
        self.conditions.blockSignals(False)
        self._reference_choices()
        import re
        matched = sum(bool(re.match(pattern, Path(path).name, re.I)) for path in all_files) if pattern else len(all_files)
        self.status.setText(tr(
            "Source: {source}. Indexed {count} fields; {matched} matched filenames and "
            "{unmatched} unmatched filenames excluded. Available channel IDs in the first field: {channels}.",
            source=self.source.text(), count=len(self._fields), matched=matched,
            unmatched=len(all_files) - matched, channels=str(available)))
        valid = bool(self._fields) and all(field.well and not field.unresolved for field in self._fields)
        self.next.setEnabled(valid)
        if not valid:
            self.status.setText(tr("The filename regex must include a valid wellID for every image."))
        self._readout_timer.start(0)

    def _fit_readouts(self):
        """Reserve wrapped metadata/status heights instead of clipping their text."""
        from ..gui_scale import _set_actual_dim
        width = max(200, self.preview.viewport().width())
        for label in (self.axes, self.status):
            _set_actual_dim(label, "minh", max(label.fontMetrics().height(), label.heightForWidth(width)))

    def resizeEvent(self, event):
        """Reflow metadata messages when the normal module pane changes width."""
        super().resizeEvent(event)
        self._widen_settings()
        self._readout_timer.start(0)

    def _settings_moved(self, *_):
        """Keep the width the user dragged the settings column to."""
        self._settings_dragged = True

    def _widen_settings(self):
        """Open the settings column :data:`_OVERVIEW_SETTINGS_WIDTH` wide.

        The splitter otherwise gives it its minimum, which cut its labels
        and choices off. At most three fifths of the step's width go to it,
        and a width the user dragged it to is kept.
        """
        split = self._wizard_content
        sizes = split.sizes()
        total = sum(sizes)
        if self._settings_dragged or len(sizes) != 2 or total <= 0:
            return
        wide = min(_OVERVIEW_SETTINGS_WIDTH, total * 3 // 5)
        if sizes[0] < wide:
            split.setSizes([wide, total - wide])

    def _choose_geometry(self):
        """Keep an explicit user geometry across subsequent metadata previews."""
        self._geometry_explicit = True
        if self.well_order.isEnabled():
            self._choose_well_order()

    def _choose_well_order(self, *_):
        """Place numeric well indices again with the chosen geometry and numbering."""
        geometry = self.plate_geometry.currentData() or (None, None)
        self._well_layout = (geometry[0], geometry[1], self.well_order.currentData())
        self._layout_explicit = True
        self._geometry_explicit = True
        self._scan()

    def _pixel_size_edited(self, *_):
        """Keep a typed pixel size across later metadata previews."""
        self._pixel_size_explicit = True

    @staticmethod
    def _acquisition_sentences(acquisition):
        """Describe what the acquisition metadata and filenames established."""
        sentences = []
        if acquisition.get("sources"):
            sentences.append(tr("Acquisition metadata from {instrument}: {sources}.",
                                instrument=acquisition.get("instrument", ""),
                                sources=", ".join(acquisition["sources"])))
        else:
            sentences.append(tr("No acquisition metadata files were found; values come from filenames and image headers."))
        sentences.append(tr("Detected {wells} wells, {fields} fields, {z} Z planes and {time} time points.",
                            wells=acquisition.get("wells", 0), fields=acquisition.get("fields") or 0,
                            z=acquisition.get("z_planes", 1), time=acquisition.get("time_points", 1)))
        if acquisition.get("numeric_wells"):
            if acquisition.get("rows") and acquisition.get("columns"):
                order = {"row": tr("row by row"), "column": tr("column by column")}.get(
                    acquisition.get("well_order"), tr("by the acquisition well table"))
                sentences.append(tr("Numeric well indices are placed on a {rows} × {columns} plate numbered {order}.",
                                    rows=acquisition["rows"], columns=acquisition["columns"], order=order))
            else:
                sentences.append(tr("Numeric well indices have no plate metadata; the CQ1 default of 24 columns "
                                    "numbered row by row is used until you choose a geometry."))
        return tuple(sentences)

    def _fill_channel_table(self, available, detected):
        """Show detected channel names, wavelengths and colours as editable rows."""
        from .sortable_table import table_item
        from ...regex_infer import _channel_display_colour
        previous = self._channel_overrides() if getattr(self, "_edited_channels", None) else {}
        self.channel_table.blockSignals(True)
        self.channel_table.setSortingEnabled(False)
        self.channel_table.setRowCount(len(available))
        for row, channel in enumerate(available):
            record = detected.get(channel) or detected.get(str(channel)) or {}
            name, colour = previous.get(channel, (record.get("name") or "",
                record.get("color") or _channel_display_colour(position=row)))
            wavelength = record.get("excitation_nm")
            wavelength = " / ".join(part for part in (
                f"{wavelength:g} nm" if wavelength else "", record.get("emission") or "") if part)
            for column, value in enumerate((str(channel), name, wavelength, colour)):
                item = table_item(str(value))
                if column in (0, 2):
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.channel_table.setItem(row, column, item)
            self._paint_colour_cell(row)
        self.channel_table.setSortingEnabled(True)
        self.channel_table.blockSignals(False)

    def _paint_colour_cell(self, row):
        """Show a channel's display colour as the colour cell's swatch."""
        from PySide6.QtGui import QColor
        item = self.channel_table.item(row, 3)
        if item is None:
            return
        colour = QColor(item.text().strip())
        if colour.isValid():
            item.setBackground(colour)
            item.setForeground(QColor("#000000") if colour.lightness() > 110 else QColor("#ffffff"))

    def _channel_colour_edited(self, item):
        """Repaint an edited colour cell and remember the user's names and colours."""
        if item.column() == 3:
            self.channel_table.blockSignals(True)
            self._paint_colour_cell(item.row())
            self.channel_table.blockSignals(False)
        self._edited_channels = self._channel_overrides()

    def _channel_overrides(self):
        """Return ``{channel: (name, colour)}`` from the table's current rows."""
        output = {}
        for row in range(self.channel_table.rowCount()):
            items = [self.channel_table.item(row, column) for column in range(4)]
            if all(item is not None for item in items) and items[0].text().isdigit():
                output[int(items[0].text())] = (items[1].text().strip(), items[3].text().strip())
        return output

    def _channel_display(self):
        """Validate the channel table into names and ``#rrggbb`` colours."""
        from PySide6.QtGui import QColor
        names, colours = {}, {}
        for channel, (name, colour) in self._channel_overrides().items():
            value = QColor(colour)
            if not value.isValid():
                raise _OverviewValidationError(tr("Channel {channel} has no valid colour: {colour}.",
                                                  channel=channel, colour=colour))
            if name:
                names[channel] = name
            colours[channel] = value.name()
        return names, colours

    def _condition_mapping(self):
        """Read explicit condition labels without changing any image metadata."""
        output = {}
        for row in range(self.conditions.rowCount()):
            items = [self.conditions.item(row, column) for column in range(3)]
            if all(item is not None for item in items) and items[2].text().strip():
                output[(items[0].text(), items[1].text())] = items[2].text().strip()
        return output

    def _reference_choices(self):
        """List only existing plate-qualified references or explicit conditions."""
        mode = self.mode.currentData()
        self.reference_kind.setEnabled(mode == "reference_average")
        if mode == "reference_average":
            mode = self.reference_kind.currentData()
        conditions = self._condition_mapping()
        values = {_overview_reference_key(field, mode, conditions) for field in self._fields}
        values.discard(None)
        values.discard("")
        previous = self.reference.currentData()
        self.reference.clear()
        for value in sorted(values, key=str):
            self.reference.addItem(str(value), value)
        index = next((index for index in range(self.reference.count())
                      if self.reference.itemData(index) == previous), -1)
        if index >= 0:
            self.reference.setCurrentIndex(index)
        self.reference.setEnabled(mode.startswith("reference_"))

    def configuration(self):
        """Return validated metadata and display choices for the browser worker."""
        if not self._info or not self.next.isEnabled() or any(not field.well or field.unresolved for field in self._fields):
            raise _OverviewValidationError(tr("Preview the current source metadata before continuing."))
        try:
            channels = tuple(int(value.strip()) for value in self.channels.text().split(","))
        except ValueError:
            raise _OverviewValidationError(tr("Choose only channel IDs reported by this image's metadata.")) from None
        if not channels or len(set(channels)) != len(channels) or not set(channels) <= set(self._info["channels"]):
            raise _OverviewValidationError(tr("Choose only channel IDs reported by this image's metadata."))
        rows_required, columns_required = _overview_geometry_extent(self._fields)
        geometry = self.plate_geometry.currentData()
        if geometry is None:
            raise _OverviewValidationError(tr("Choose a plate geometry."))
        rows_available, columns_available = geometry
        if rows_required > rows_available or columns_required > columns_available:
            raise _OverviewValidationError(tr(
                "The selected geometry does not contain the indexed wells "
                "({rows} rows, {columns} columns required).",
                rows=rows_required, columns=columns_required))
        if not self.low.value() < self.high.value():
            raise _OverviewValidationError(tr("The lower percentile must be below the upper percentile."))
        mode = self.mode.currentData()
        reference = self.reference.currentData()
        if mode.startswith("reference_") and reference is None:
            raise _OverviewValidationError(tr("The selected normalization reference has no fields."))
        gain = self.gain.currentData()
        if mode == "reference_average" and (gain is not None or self.mean_match.isChecked()):
            raise _OverviewValidationError(tr(
                "Well-mean display scaling cannot be combined with channel-reference scaling."))
        if self.mean_match.isChecked() and gain is None:
            raise _OverviewValidationError(tr("Choose a reference channel before enabling mean-matched display scaling."))
        if gain is not None and gain not in channels:
            raise _OverviewValidationError(tr("The reference channel must be selected for display."))
        channel_names, channel_colors = self._channel_display()
        return {"source": self.source.text(), "custom_regex": self.regex.text(),
                "plate_names": dict(self._plate_names), "use_subfolders": tuple(sorted(self._use_subfolders)),
                "plate_regex": dict(self._plate_regex),
                "excluded_plates": tuple(sorted(self._excluded_plates)),
                "well_layout": tuple(self._well_layout) if self._layout_explicit else None,
                "channel_names": channel_names, "channel_colors": channel_colors,
                "pixel_size_um": self.pixel_size.value() or None,
                "acquisition": dict((self._info or {}).get("acquisition") or {}),
                "microscope": self.preset.currentData(), "source_scope": self.scope.currentData(),
                "geometry": tuple(self.plate_geometry.currentData()),
                "channels": channels, "time_index": self.time.value(),
                "z_mode": self.z_mode.currentData(), "z_index": self.z.value(),
                "representatives": self.representatives.value(),
                "normalize": bool(self._settings.get("normalize", True)),
                "normalization": {"mode": mode, "percentiles": (self.low.value(), self.high.value()),
                                  "reference": reference, "reference_kind": self.reference_kind.currentData(),
                                  "reference_channel": gain,
                                  "mean_match": self.mean_match.isChecked(),
                                  "conditions": self._condition_mapping()}}

    def _next(self):
        """Accept only a valid current configuration; report errors in place."""
        try:
            self.configuration()
        except _OverviewValidationError as error:
            self.status.setText(str(error))
            return
        self.accept()

    def _failed(self, error):
        """Report worker refusals while leaving the source and raw files intact."""
        self.status.setText(str(error))
        self.next.setEnabled(False)

    def result(self):
        """Return the accepted or rejected metadata result."""
        return self._result

    def accept(self):
        """Publish validated metadata without turning the body into a popup."""
        self.done(QDialog.Accepted)

    def reject(self):
        """Cancel this metadata generation and notify its embedded workspace."""
        self.done(QDialog.Rejected)

    def done(self, result):
        """Cancel metadata work before hiding and emitting the matching result."""
        if self._finished:
            return
        self._finished = True
        self._result = result
        self._startup_timer.stop()
        self._rescan_timer.stop()
        self._readout_timer.stop()
        self._cancelled.set()
        self._action_fit_timer.stop()
        self._runner.shutdown(timeout_ms=0)
        self.hide()
        if result == QDialog.Accepted:
            self.accepted.emit()
        else:
            self.rejected.emit()
        self.finished.emit(result)
        from ..gil_priority import release_background_imports

        release_background_imports(self)

    def closeEvent(self, event):
        """Retire pending metadata when its containing window is closed."""
        self.reject()
        super().closeEvent(event)

    def keyPressEvent(self, event):
        """Retain Escape cancellation inside the metadata body."""
        if event.key() == Qt.Key_Escape:
            self.reject()
            event.accept()
            return
        super().keyPressEvent(event)


def _overview_prepare_normalization(fields, channels, options, read_planes, cancelled=None):
    """Return a JSON display plan and a separate optional lazy well-statistics context."""
    if (options or {}).get("mode") == "reference_average":
        context = _OverviewWellMeanContext(fields, channels, options, read_planes, cancelled)
        return context.provenance(), context
    return _overview_normalization_plan(fields, channels, options, read_planes, cancelled), None


def _overview_sample_mean(values):
    """Compute a finite-sample mean without overflowing finite intensity magnitudes."""
    scale = float(np.max(np.abs(values)))
    return float(np.mean(values / scale) * scale) if scale else 0.


def _overview_cancelled(cancelled):
    """Stop statistics at a field or channel boundary when cancellation is requested."""
    if cancelled is not None and cancelled.is_set():
        from concurrent.futures import CancelledError
        raise CancelledError()


class _OverviewWellMeanContext:
    """Learn equal-field/equal-well display gains without retaining image arrays."""

    def __init__(self, fields, channels, options, read_planes, cancelled=None):
        """Learn the chosen same-plate reference and bounded original-pixel bounds."""
        from collections import OrderedDict
        import threading

        options = dict(options)
        if options.get("reference_channel") is not None or options.get("mean_match", False):
            raise _OverviewValidationError(tr(
                "Well-mean display scaling cannot be combined with channel-reference scaling."))
        self.channels = tuple(channels)
        checked = _overview_normalization_plan((), self.channels,
            {"mode": "image", "percentiles": options.get("percentiles", (1., 99.))}, None)
        self._percentiles = checked["percentiles"]
        self._reference_kind = options.get("reference_kind", "reference_column")
        if self._reference_kind not in ("reference_well", "reference_row", "reference_column",
                                       "reference_condition"):
            raise _OverviewValidationError(tr("Choose a supported normalization mode."))
        reference = options.get("reference")
        if not isinstance(reference, (list, tuple)) or len(reference) != 2:
            raise _OverviewValidationError(tr("The selected normalization reference has no fields."))
        self._reference = tuple(reference)
        self._plate = self._reference[0]
        self._selection = dict(options.get("selection") or {})
        self._read_planes = read_planes
        self._cancelled = cancelled
        self._lock = threading.RLock()
        self._cache = OrderedDict()
        self._pending = {}
        self._used = {}
        self._fields = {}
        selected = tuple(field for field in fields if field.plate == self._plate)
        if not selected:
            raise _OverviewValidationError(tr("The selected normalization reference has no fields."))
        if (len({getattr(field, "time_id", "") for field in selected}) > 1
                or len({getattr(field, "z_id", "") for field in selected}) > 1):
            raise _OverviewValidationError(tr(
                "Choose one filename time point and Z plane before well-mean display scaling."))
        for field in selected:
            if not field.well or not field.row or field.column <= 0:
                raise _OverviewValidationError(tr(
                    "Well-mean display scaling requires resolved physical well identities."))
            key = (field.plate, field.well)
            self._fields.setdefault(key, {})[field.identity] = field
        self._fields = {key: tuple(group.values()) for key, group in self._fields.items()}
        conditions = dict(options.get("conditions") or {})
        self._reference_keys = tuple(sorted({(field.plate, field.well) for field in selected
            if _overview_reference_key(field, self._reference_kind, conditions) == self._reference}))
        if not self._reference_keys:
            raise _OverviewValidationError(tr("The selected normalization reference has no fields."))
        budget = 262144 // len(self._reference_keys)
        if budget < 1:
            raise _OverviewValidationError(tr("The reference group is too large for bounded sampling."))
        pools = {channel: [] for channel in self.channels}
        self._reference_stats = {}
        self._bounds_counts = {}
        for key in self._reference_keys:
            _overview_cancelled(cancelled)
            stats, values = self._learn_well(key, cancelled, budget)
            self._reference_stats[key] = stats
            for channel in self.channels:
                pools[channel].append(values[channel])
        for channel in self.channels:
            count = min(len(values) for values in pools[channel])
            pools[channel] = [values[np.linspace(0, len(values) - 1, count, dtype=int)]
                              for values in pools[channel]]
            for key in self._reference_keys:
                self._bounds_counts.setdefault(key, {})[channel] = count
        self._reference_means = {channel: _overview_sample_mean(np.array([
            self._reference_stats[key]["means"][channel] for key in self._reference_keys]))
            for channel in self.channels}
        self._bounds = {channel: tuple(float(value) for value in np.percentile(
            np.concatenate(pools[channel]), self._percentiles)) for channel in self.channels}

    def _new_generation(self, cancelled):
        """Fork completed scalar statistics with independent cancellation and pending reads.

        Immutable field identities are retained; images, in-flight futures and
        provenance usage are never transferred to the fresh generation.
        """
        import copy
        import threading
        from collections import OrderedDict

        with self._lock:
            fresh = copy.copy(self)
            fresh._cancelled = cancelled
            fresh._lock = threading.RLock()
            fresh._pending = {}
            fresh._used = {}
            fresh._fields = dict(self._fields)
            fresh._selection = dict(self._selection)
            fresh._bounds = dict(self._bounds)
            fresh._reference_means = dict(self._reference_means)
            fresh._bounds_counts = {key: dict(counts) for key, counts in self._bounds_counts.items()}
            fresh._reference_stats = {key: {**stats, "means": dict(stats["means"]),
                "sample_pixels": dict(stats["sample_pixels"])}
                for key, stats in self._reference_stats.items()}
            fresh._cache = OrderedDict((key, {**stats, "means": dict(stats["means"]),
                "sample_pixels": dict(stats["sample_pixels"])}) for key, stats in self._cache.items())
        return fresh

    def _learn_well(self, key, cancelled, bounds_budget=0):
        """Average every selected FOV equally; optionally retain a bounded raw reservoir."""
        import hashlib
        import json

        fields = self._fields[key]
        field_means = {channel: [] for channel in self.channels}
        sample_counts = {channel: 0 for channel in self.channels}
        retained = {channel: np.empty(0, dtype=np.float64) for channel in self.channels}
        priorities = {channel: np.empty(0, dtype=np.float64) for channel in self.channels}
        rng = {}
        if bounds_budget:
            for channel in self.channels:
                digest = hashlib.sha256(json.dumps([key, channel, self._selection],
                    sort_keys=True).encode("utf-8")).digest()
                rng[channel] = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        for field in fields:
            _overview_cancelled(self._cancelled)
            _overview_cancelled(cancelled)
            found = set()
            for channel, path, raw in self._read_planes(field, self.channels, edge=256):
                _overview_cancelled(self._cancelled)
                _overview_cancelled(cancelled)
                if channel not in field_means or channel in found:
                    raise _OverviewValidationError(tr("The normalization sample has ambiguous channels."))
                found.add(channel)
                values = np.asarray(raw).ravel()
                values = values[np.isfinite(values)]
                if not values.size:
                    raise _OverviewValidationError(tr("This channel has no finite pixels."))
                positions = np.linspace(0, values.size - 1, min(2048, values.size), dtype=int)
                values = np.asarray(values[positions], dtype=np.float64)
                field_means[channel].append(_overview_sample_mean(values))
                sample_counts[channel] += len(values)
                if bounds_budget:
                    retained[channel] = np.concatenate((retained[channel], values))
                    priorities[channel] = np.concatenate((priorities[channel], rng[channel].random(len(values))))
                    if len(retained[channel]) > bounds_budget:
                        chosen = np.argpartition(priorities[channel], bounds_budget - 1)[:bounds_budget]
                        retained[channel] = retained[channel][chosen]
                        priorities[channel] = priorities[channel][chosen]
            if found != set(self.channels):
                raise _OverviewValidationError(tr("A selected channel is missing from the normalization sample."))
        means = {channel: _overview_sample_mean(np.asarray(values))
                 for channel, values in field_means.items()}
        if any(not math.isfinite(value) or value <= 0 for value in means.values()):
            raise _OverviewValidationError(tr(
                "Well-mean display scaling requires positive finite means for every selected well and channel."))
        return {"means": means, "sample_pixels": sample_counts,
                "field_identities": tuple(field.identity for field in fields)}, retained

    def _well_stats(self, key, cancelled):
        """Coalesce concurrent reads and keep at most 128 nonreference scalar-stat records."""
        from concurrent.futures import Future, TimeoutError

        _overview_cancelled(self._cancelled)
        _overview_cancelled(cancelled)
        with self._lock:
            if key in self._reference_stats:
                return self._reference_stats[key]
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            future = self._pending.get(key)
            owns_read = future is None
            if owns_read:
                future = Future()
                self._pending[key] = future
        if not owns_read:
            while True:
                _overview_cancelled(self._cancelled)
                _overview_cancelled(cancelled)
                try:
                    return future.result(timeout=.1)
                except TimeoutError:
                    continue
        try:
            stats, values = self._learn_well(key, cancelled)
        except BaseException as error:
            with self._lock:
                self._pending.pop(key, None)
                future.set_exception(error)
            raise
        with self._lock:
            self._cache[key] = stats
            self._cache.move_to_end(key)
            while len(self._cache) > 128:
                self._cache.popitem(last=False)
            self._pending.pop(key, None)
            future.set_result(stats)
        return stats

    def plan_for(self, field, cancelled=None):
        """Resolve this well's gain on a worker without changing raw measurements."""
        key = (field.plate, field.well)
        if key not in self._fields:
            raise _OverviewValidationError(tr(
                "This field is outside the selected same-plate reference and representative scope."))
        stats = self._well_stats(key, cancelled)
        gains = {channel: self._reference_means[channel] / stats["means"][channel]
                 for channel in self.channels}
        if any(not math.isfinite(value) or value <= 0 for value in gains.values()):
            raise _OverviewValidationError(tr(
                "Well-mean display scaling requires positive finite means for every selected well and channel."))
        with self._lock:
            self._used[key] = {"plate": key[0], "well": key[1], **stats, "gains": gains}
        return {"mode": "reference_average", "percentiles": self._percentiles,
                "bounds": dict(self._bounds), "gains": gains, "well_identity": key}

    def provenance(self):
        """Return JSON-safe means, sampling choices and used gains, excluding runtime caches."""
        with self._lock:
            reference = [{"plate": key[0], "well": key[1], **self._reference_stats[key],
                          "bounds_sample_pixels": self._bounds_counts[key]}
                         for key in self._reference_keys]
            used = [dict(self._used[key]) for key in sorted(self._used)]
        return {"mode": "reference_average", "reference_kind": self._reference_kind,
                "reference": self._reference, "scope_plate": self._plate,
                "percentiles": self._percentiles, "bounds": dict(self._bounds),
                "reference_means": dict(self._reference_means),
                "reference_wells": reference, "used_wells": used,
                "selection": dict(self._selection), "preview_edge": 256,
                "max_pixels_per_field": 2048, "max_bounds_pixels_per_channel": 262144,
                "bounds_sampling": "balanced wells; deterministic PCG64 priority reservoir",
                "bounds_domain": "original reference pixels before gain", "nonreference_cache_wells": 128}


from ..theme import register_widget_qss as _register_widget_qss

_register_widget_qss("FieldOverviewWorkspace", _overview_workspace_qss, replace=True)
_register_widget_qss("FieldOverviewPanel", _overview_panel_qss, replace=True)


from PySide6.QtCore import QObject as _QObject

_OVERVIEW_CANVAS_SOURCE = "overview_field"
_OVERVIEW_GUIDED_SIDES = ("none", "above", "below", "left", "right")
_OVERVIEW_GUIDED_ITEMS = ("plate", "well", "condition", "field", "channels", "measurements", "text")
_OVERVIEW_GUIDED_DEFAULT_SIDES = {"plate": "above", "well": "above", "condition": "above"}
_OVERVIEW_GUIDED_MARGIN = 36.0
_OVERVIEW_GUIDED_GAP = 10.0
_OVERVIEW_GUIDED_IMAGE_GAP = 4.0
_OVERVIEW_GUIDED_FONT = 9.0
_OVERVIEW_GUIDED_SIDE_WIDTH = 108.0
_OVERVIEW_GUIDED_MAX_WELL_FIELDS = 24
_OVERVIEW_GUIDED_CAPTIONS = {
    "plate": "Plate", "well": "Well", "condition": "Condition", "field": "Field",
    "channels": "Channel names", "measurements": "Measurements", "text": "Free text"}
_OVERVIEW_GUIDED_SIDE_CAPTIONS = {"none": "Not shown", "above": "Above", "below": "Below",
                                  "left": "Left", "right": "Right"}
_OVERVIEW_GUIDED_WELL_MODES = (
    ("By condition", "Every well whose annotation or comparison group matches a ticked condition."),
    ("Choose wells", "Tick the wells to export."),
    ("One well", "Export a single well."),
)
_OVERVIEW_GUIDED_FIELD_MODES = (
    ("All fields", "Every field of each chosen well at the shown time and Z."),
    ("First fields", "The first fields of each well, as many as set."),
    ("Pick fields", "Tick the fields to export in each well."),
)


def _overview_canvas_image(field, pixel_size_um=None):
    """The canvas image entry of one Overview field.

    :param field: an :class:`_OverviewField`.
    :param pixel_size_um: the field's pixel size, when known.
    :returns: a dictionary for :meth:`_CanvasComposer.add_images`.
    """
    return {"source": {"type": _OVERVIEW_CANVAS_SOURCE, "key": list(_overview_image_key(field)),
                       "plate": field.plate, "well": field.well, "field": field.field},
            "label": field.caption, "pixel_size_um": float(pixel_size_um) if pixel_size_um else None}


def _overview_file_field(path):
    """An image file from disk as an :class:`_OverviewField` the display pipeline can read.

    :param path: a PNG, JPEG, BMP, TIFF or NumPy file; its channels are the
        file's own channel axis.
    """
    return _OverviewField(plate="", well="", row="", column=0, field=os.path.basename(str(path)),
                          files=(("", str(path)),), source=str(path))


def _overview_file_channel_count(path):
    """How many channels an image file has, read from its header only."""
    return max(1, len(_overview_channel_info(_overview_file_field(path))["channels"]))


def _overview_import_chain(entry):
    """The enhancement chain one channel's import settings ask for, or ``None``."""
    from ..detect_chain import Chain

    chain = Chain(background="rolling_ball" if entry.get("background") else "none",
                  denoise="gaussian" if entry.get("denoise") else "none",
                  gamma=float(entry.get("gamma") or 1.0), clahe=bool(entry.get("clahe")),
                  sharpen=bool(entry.get("sharpen")))
    return None if chain == Chain() else chain


def _overview_import_pixels(box, edge, display=None):
    """An image file placed on the export canvas, through :func:`_overview_pixels`.

    The box's import settings (``box["import"]["channels"]``, one entry per
    channel; see ``picture_export._CANVAS_IMPORT_DEFAULTS``) choose which
    channels are drawn and set each one's percentile normalization and its
    enhancement chain, as the Overview does for its own images; the
    Overview's colour mode applies too.

    :param box: an image box whose source is a file.
    :param edge: longest side wanted, or ``None`` for every source pixel.
    :param display: the Overview's display keywords; only ``color_mode`` is used.
    :returns: an 8-bit RGB array.
    """
    from .picture_export import _canvas_import_entry

    field = _overview_file_field((box.get("source") or {}).get("path", ""))
    available = tuple(_overview_channel_info(field)["channels"])
    entries = {channel: _canvas_import_entry(box, channel) for channel in available}
    shown = tuple(channel for channel in available if entries[channel]["shown"]) or available[:1]
    normalized = [channel for channel in shown if entries[channel]["normalize"]]
    pixels = _overview_pixels(
        field, shown, (1., 99.9), edge=edge, percentile_channels=normalized,
        channel_percentiles={channel: (float(entries[channel]["low"]), float(entries[channel]["high"]))
                             for channel in normalized},
        channel_enhancement={channel: _overview_import_chain(entries[channel]) for channel in shown},
        color_mode=str((display or {}).get("color_mode") or "additive"))
    return np.ascontiguousarray(pixels[..., :3] if pixels.ndim == 3 else pixels)


class _OverviewCanvasSnapshot:
    """What the browser shows, frozen on the GUI thread, for rendering fields elsewhere.

    Images go through :func:`_overview_pixels` with the browser's display
    keywords, its scope limits and per-image overrides, so a canvas image
    matches the screen; the display-only enhancement chain stays out, as in
    every export. Image files with import settings go through the same
    function (:func:`_overview_import_pixels`).
    """

    imports = True

    def __init__(self, fields, channels, limits, display):
        """Keep the fields by image key and the display settings."""
        self._fields = {_overview_key_text(_overview_image_key(field)): field for field in fields}
        self._channels = tuple(channels)
        self._limits = limits
        self._display = dict(display)

    def field_of(self, box):
        """The field an image box shows, or ``None``."""
        source = box.get("source") or {}
        if source.get("type") != _OVERVIEW_CANVAS_SOURCE:
            return None
        return self._fields.get(_overview_key_text(tuple(source.get("key") or ())))

    def __call__(self, box, edge):
        """The box's field rendered with the frozen display settings.

        :param edge: longest side wanted, or ``None`` for every source pixel.
        :raises ValueError: when the field is not in the loaded plates.
        """
        if (box.get("source") or {}).get("type") == "file":
            return _overview_import_pixels(box, edge, self._display)
        field = self.field_of(box)
        if field is None:
            raise ValueError(tr("The image source is not loaded: {label}", label=box.get("label") or "?"))
        pixels = _overview_pixels(field, self._channels, self._limits, edge=edge, **self._display)
        return np.ascontiguousarray(pixels[..., :3] if pixels.ndim == 3 else pixels)


class _OverviewCanvasSource:
    """Hands the composer a fresh snapshot of the browser's display whenever it renders."""

    imports = True

    def __init__(self, browser):
        """Follow ``browser`` without keeping it alive."""
        import weakref

        self._browser = weakref.ref(browser)

    def fresh(self):
        """A :class:`_OverviewCanvasSnapshot` of the browser now, taken on the GUI thread."""
        browser = self._browser()
        if browser is None:
            return _OverviewCanvasSnapshot((), (0,), None, {})
        return browser._canvas_snapshot()

    @staticmethod
    def file_channels(path):
        """How many channels an image file has, for its import settings."""
        return _overview_file_channel_count(path)


_OVERVIEW_CANVAS_HOLD_MS = 350
_OVERVIEW_EXPORT_DOCK_MIN_WIDTH = 600


class _OverviewExportDock(QFrame):
    """The export section docked at the far right of the Overview: a heading, Close and the export itself.

    :param title: the heading, already translated.
    :param content: the Guided export or the Custom export canvas.
    :param hint: an optional line under the heading, already translated.
    """

    closed = Signal()

    def __init__(self, title, content, hint=""):
        """Lay out the heading row, the hint and ``content``."""
        super().__init__()
        self.setObjectName("OverviewExportDock")
        self.setFrameShape(QFrame.StyledPanel)
        self.content = content
        column = QVBoxLayout(self)
        column.setContentsMargins(6, 6, 6, 6)
        column.setSpacing(4)
        header = QHBoxLayout()
        self.title = QLabel(title)
        self.title.setObjectName("OverviewExportDockTitle")
        font = self.title.font()
        font.setBold(True)
        self.title.setFont(font)
        self.close_button = QPushButton(tr("Close"))
        self.close_button.setObjectName("OverviewExportClose")
        self.close_button.clicked.connect(self.closed)
        header.addWidget(self.title, 1)
        header.addWidget(self.close_button)
        column.addLayout(header)
        self.hint = QLabel(hint)
        self.hint.setObjectName("OverviewExportDockHint")
        self.hint.setWordWrap(True)
        self.hint.setVisible(bool(hint))
        column.addWidget(self.hint)
        content.setWindowFlags(Qt.Widget)
        column.addWidget(content, 1)
        content.show()


class _OverviewCanvasDrags(_QObject):
    """Bring Overview images to the Custom export canvas while it is open.

    Every way works from the plate tiles, the well views and the image views:

    - press on a selected well or field and move: the selection is dragged;
    - press and hold any image for :data:`_OVERVIEW_CANVAS_HOLD_MS`, then
      move: that image (or the selection it belongs to) is dragged, while a
      plain drag still pans an image or selects wells;
    - drag the "Drag to canvas" handle in an image view's corner, or drag
      the shown image with Alt held;
    - pick mode: hold Ctrl+Shift and click any image to add it straight away.

    The filter sits on the application while the canvas is open, so wells
    and images opened afterwards take part too.
    """

    def __init__(self, browser):
        """Follow ``browser``'s views."""
        super().__init__(browser)
        self._browser = browser
        self._press = None
        self._watched = []
        self._handles = {}
        self._armed = False
        self._hold = QTimer(self)
        self._hold.setSingleShot(True)
        self._hold.setInterval(_OVERVIEW_CANVAS_HOLD_MS)
        self._hold.timeout.connect(self._arm)

    def watch(self):
        """Start listening, and put a drag handle on every image view shown now."""
        from PySide6.QtWidgets import QApplication

        application = QApplication.instance()
        if application is not None and application not in self._watched:
            application.installEventFilter(self)
            self._watched.append(application)
        self._sync_handles()

    def unwatch(self, *_):
        """Stop listening and remove the drag handles."""
        from shiboken6 import isValid

        for watched in self._watched:
            if isValid(watched):
                watched.removeEventFilter(self)
        self._watched = []
        self._hold.stop()
        self._press = None
        self._armed = False
        for handle in list(self._handles):
            if isValid(handle):
                handle.hide()
                handle.deleteLater()
        self._handles = {}

    def _image_viewports(self):
        """The viewports of the enlarged image and of every stacked image."""
        browser = self._browser
        return [browser.view.viewport()] + [panel.view.viewport() for panel in browser._image_views]

    def _viewports(self):
        """Every viewport an image can be taken from."""
        browser = self._browser
        return ([table.viewport() for table in browser._plate_table_list()]
                + [field_list.viewport() for field_list in browser._field_lists()]
                + self._image_viewports())

    def _sync_handles(self):
        """Give each image view a handle in its top-right corner; drop handles of closed views."""
        from shiboken6 import isValid

        viewports = self._image_viewports()
        for handle, viewport in list(self._handles.items()):
            if not isValid(handle) or not isValid(viewport) or viewport not in viewports:
                self._handles.pop(handle, None)
                if isValid(handle):
                    handle.deleteLater()
        known = set(self._handles.values())
        for viewport in viewports:
            if viewport in known:
                continue
            handle = QToolButton(viewport)
            handle.setObjectName("OverviewCanvasHandle")
            handle.setText(tr("Drag to canvas"))
            handle.setCursor(Qt.OpenHandCursor)
            handle.setToolTip(tr("Drag this image onto the export canvas. You can also press and hold any image "
                                 "and then drag it, or hold {keys} and click an image to add it.",
                                 keys=_overview_pick_keys()))
            handle.adjustSize()
            self._handles[handle] = viewport
            handle.show()
        for handle, viewport in self._handles.items():
            handle.move(max(0, viewport.width() - handle.width() - 6), 6)
            handle.raise_()

    def _images_at(self, viewport, position, alt, any_image=False):
        """The canvas images a press at ``position`` would bring, or ``[]``.

        :param alt: whether Alt is held, which lets the shown image of an
            image view drag at once.
        :param any_image: also take an unselected well, field or image (press
            and hold, pick mode); otherwise only a selection, or an image
            view's image with ``alt``.
        """
        browser = self._browser
        widget = viewport.parentWidget()
        pixel_size = browser._overview_options.get("pixel_size_um")
        if widget in browser._plate_table_list():
            item = widget.itemAt(position)
            if item is None or not (item.isSelected() or any_image):
                return []
            plate = widget._overview_plate
            positions = {value: address for address, value in browser._well_positions.items() if address[0] == plate}
            fields = []
            for cell in (widget.selectedItems() if item.isSelected() else [item]):
                address = positions.get((cell.row(), cell.column()))
                if address is not None:
                    fields.extend(browser._canvas_well_fields(*address))
            return [_overview_canvas_image(field, pixel_size) for field in fields[:_OVERVIEW_GUIDED_MAX_WELL_FIELDS]]
        if widget in browser._field_lists():
            item = widget.itemAt(position)
            if item is None or not (item.isSelected() or any_image):
                return []
            chosen = widget.selectedItems() if item.isSelected() else [item]
            fields = [entry.data(Qt.UserRole) for entry in chosen]
            return [_overview_canvas_image(field, pixel_size) for field in fields if field is not None]
        if not (alt or any_image):
            return []
        field = browser._field_shown_in(widget)
        return [] if field is None else [_overview_canvas_image(field, pixel_size)]

    def _arm(self):
        """The press was held: the next move drags the image."""
        if self._press is None:
            return
        self._armed = True
        self._press[0].setCursor(Qt.DragCopyCursor)
        composer = getattr(self._browser, "_canvas_composer", None)
        if composer is not None:
            composer.status.setText(tr("Move the pointer to drag the image onto the canvas."))

    def _disarm(self):
        """Forget the press and restore the cursor."""
        from shiboken6 import isValid

        self._hold.stop()
        if self._press is not None and self._armed and isValid(self._press[0]):
            self._press[0].unsetCursor()
        self._press = None
        self._armed = False

    def _run_drag(self, drag):
        """Run the drag until it is dropped or abandoned."""
        return drag.exec(Qt.CopyAction)

    def _start_drag(self, watched, images, position):
        """Drag ``images`` from ``watched``; returns the drop action.

        A view that saw the press gets a release afterwards, so it does not
        go on panning or selecting once the drag is over.
        """
        from PySide6.QtGui import QDrag, QMouseEvent
        from PySide6.QtWidgets import QApplication
        from shiboken6 import isValid
        from .picture_export import _canvas_mime

        drag = QDrag(watched)
        drag.setMimeData(_canvas_mime(images))
        result = self._run_drag(drag)
        if isValid(watched) and watched not in self._handles:
            release = QMouseEvent(QEvent.MouseButtonRelease, position.toPointF(),
                                  watched.mapToGlobal(position).toPointF(),
                                  Qt.LeftButton, Qt.NoButton, Qt.NoModifier)
            QApplication.sendEvent(watched, release)
        return result

    def eventFilter(self, watched, event):
        """Start canvas drags and picks from presses on the Overview's images."""
        from PySide6.QtWidgets import QApplication

        kind = event.type()
        if kind not in (QEvent.MouseButtonPress, QEvent.MouseMove, QEvent.MouseButtonRelease,
                        QEvent.Enter, QEvent.Resize):
            return False
        if watched in self._handles:
            return self._handle_event(watched, event)
        if kind in (QEvent.Enter, QEvent.Resize):
            if watched in self._image_viewports():
                self._sync_handles()
            return False
        if kind == QEvent.MouseButtonPress:
            if event.button() != Qt.LeftButton or watched not in self._viewports():
                return False
            self._disarm()
            position = event.position().toPoint()
            modifiers = event.modifiers()
            if modifiers & Qt.ControlModifier and modifiers & Qt.ShiftModifier:
                images = self._images_at(watched, position, False, any_image=True)
                if images:
                    self._browser._canvas_pick(images)
                    return True
                return False
            alt = bool(modifiers & Qt.AltModifier)
            images = self._images_at(watched, position, alt)
            if images:
                self._press = (watched, position, images, True)
                return alt
            images = self._images_at(watched, position, False, any_image=True)
            if images:
                self._press = (watched, position, images, False)
                self._hold.start()
            return False
        if kind == QEvent.MouseMove and self._press is not None and self._press[0] is watched:
            start, images, ready = self._press[1], self._press[2], self._press[3]
            moved = (event.position().toPoint() - start).manhattanLength() >= QApplication.startDragDistance()
            if ready or self._armed:
                if not moved:
                    return True
                self._disarm()
                self._start_drag(watched, images, start)
                return True
            if moved:
                self._disarm()
            return False
        if kind == QEvent.MouseButtonRelease and self._press is not None:
            self._disarm()
        return False

    def _handle_event(self, handle, event):
        """Drag the image of the view a handle sits on."""
        from PySide6.QtCore import QPoint
        from PySide6.QtWidgets import QApplication

        kind = event.type()
        if kind == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            viewport = self._handles[handle]
            images = self._images_at(viewport, QPoint(0, 0), False, any_image=True)
            self._press = (handle, event.position().toPoint(), images, True) if images else None
            return True
        if kind == QEvent.MouseMove and self._press is not None and self._press[0] is handle:
            if (event.position().toPoint() - self._press[1]).manhattanLength() < QApplication.startDragDistance():
                return True
            images, start = self._press[2], self._press[1]
            self._press = None
            self._start_drag(handle, images, start)
            return True
        if kind == QEvent.MouseButtonRelease:
            self._press = None
            return True
        return False


def _overview_pick_keys():
    """The keys held to add an image by clicking it, as the keyboard shows them."""
    from .picture_export import _CANVAS_PICK_KEYS

    return _CANVAS_PICK_KEYS


def _overview_guided_lines(text, width):
    """How many caption lines ``text`` needs in a band ``width`` points wide."""
    per_line = max(4, int(width / (_OVERVIEW_GUIDED_FONT * .5)))
    return sum(max(1, math.ceil(len(line) / per_line)) for line in str(text).split("\n"))


def _overview_guided_cell_rect(layout, rows, columns, cell):
    """A placement's rectangle on the page, in points.

    :param cell: ``(row, column, row span, column span)``.
    :returns: ``(x, y, width, height)``.
    """
    width, height = layout.page_points
    cell_width = (width - 2 * _OVERVIEW_GUIDED_MARGIN - (columns - 1) * _OVERVIEW_GUIDED_GAP) / columns
    cell_height = (height - 2 * _OVERVIEW_GUIDED_MARGIN - (rows - 1) * _OVERVIEW_GUIDED_GAP) / rows
    row, column, row_span, column_span = cell
    row, column = min(row, rows - 1), min(column, columns - 1)
    row_span, column_span = max(1, min(row_span, rows - row)), max(1, min(column_span, columns - column))
    return (_OVERVIEW_GUIDED_MARGIN + column * (cell_width + _OVERVIEW_GUIDED_GAP),
            _OVERVIEW_GUIDED_MARGIN + row * (cell_height + _OVERVIEW_GUIDED_GAP),
            column_span * cell_width + (column_span - 1) * _OVERVIEW_GUIDED_GAP,
            row_span * cell_height + (row_span - 1) * _OVERVIEW_GUIDED_GAP)


def _overview_guided_layout(entries, page_size="A4", orientation="portrait", rows=3, columns=2, note=()):
    """The canvas layout of a guided export.

    :param entries: one per well, in order: ``page``, ``cell`` (row,
        column, row span, column span on the page grid), ``images`` (canvas
        image entries) and ``captions`` (``{side: text}`` for ``above``,
        ``below``, ``left`` and ``right``).
    :param rows: rows of the page grid.
    :param columns: columns of the page grid.
    :param note: lines describing the display settings, written on a last
        page when given.
    :returns: a :class:`_CanvasLayout` whose undo history starts empty.
    """
    from .picture_export import _CanvasLayout

    layout = _CanvasLayout(page_size, orientation)
    layout.snap = False
    pages = max([int(entry["page"]) + 1 for entry in entries] or [1])
    layout.pages = [[] for _ in range(pages)]
    band = _OVERVIEW_GUIDED_FONT * 1.2
    for entry in entries:
        page = int(entry["page"])
        x, y, width, height = _overview_guided_cell_rect(layout, rows, columns, entry["cell"])
        captions = {side: str(text) for side, text in (entry.get("captions") or {}).items() if str(text or "").strip()}
        side_width = min(width * .3, _OVERVIEW_GUIDED_SIDE_WIDTH)
        left = side_width if "left" in captions else 0.0
        right = side_width if "right" in captions else 0.0
        inner = width - left - right
        above = (_overview_guided_lines(captions["above"], inner) * band + 4) if "above" in captions else 0.0
        below = (_overview_guided_lines(captions["below"], inner) * band + 4) if "below" in captions else 0.0
        for side, rect in (("above", (x + left, y, inner, above)), ("below", (x + left, y + height - below, inner, below)),
                           ("left", (x, y + above, left, height - above - below)),
                           ("right", (x + width - right, y + above, right, height - above - below))):
            if side in captions:
                layout.add_box(page, "text", *rect, text=captions[side], font_size=_OVERVIEW_GUIDED_FONT,
                               align={"left": "right", "right": "left"}.get(side, "center"), role=side)
        images = list(entry.get("images") or ())
        if not images:
            continue
        area_x, area_y = x + left, y + above
        area_width, area_height = inner, height - above - below
        image_columns = max(1, math.ceil(math.sqrt(len(images) * max(area_width, 1.) / max(area_height, 1.))))
        image_columns = min(image_columns, len(images))
        image_rows = math.ceil(len(images) / image_columns)
        tile_width = (area_width - (image_columns - 1) * _OVERVIEW_GUIDED_IMAGE_GAP) / image_columns
        tile_height = (area_height - (image_rows - 1) * _OVERVIEW_GUIDED_IMAGE_GAP) / image_rows
        for index, image in enumerate(images):
            row, column = divmod(index, image_columns)
            layout.add_box(page, "image", area_x + column * (tile_width + _OVERVIEW_GUIDED_IMAGE_GAP),
                           area_y + row * (tile_height + _OVERVIEW_GUIDED_IMAGE_GAP), tile_width, tile_height,
                           fit=True, lock_aspect=False, **image)
    if note:
        width, height = layout.page_points
        layout.pages.append([])
        layout.add_box(len(layout.pages) - 1, "text", _OVERVIEW_GUIDED_MARGIN, _OVERVIEW_GUIDED_MARGIN,
                       width - 2 * _OVERVIEW_GUIDED_MARGIN, height - 2 * _OVERVIEW_GUIDED_MARGIN,
                       text="\n".join(note), font_size=8.0, role="note")
    layout.snap = True
    layout._undo.clear()
    return layout


def _overview_free_slot(occupied, rows, columns, span, page=0):
    """The first place a ``span`` block fits, scanning pages from ``page`` in reading order.

    :param occupied: ``{page: set of (row, column)}``.
    :param span: ``(row span, column span)``.
    :returns: ``(page, (row, column, row span, column span))``.
    """
    row_span, column_span = max(1, min(span[0], rows)), max(1, min(span[1], columns))
    while True:
        taken = occupied.get(page, set())
        for row in range(rows - row_span + 1):
            for column in range(columns - column_span + 1):
                block = {(row + r, column + c) for r in range(row_span) for c in range(column_span)}
                if not block & taken:
                    return page, (row, column, row_span, column_span)
        page += 1


class _OverviewCellPicker(QWidget):
    """A page preview whose grid cells are clicked or dragged over to place a well.

    Cells already given to earlier wells are shaded and labelled.
    """

    changed = Signal()

    def __init__(self, parent=None):
        """An empty 3 by 2 portrait page."""
        super().__init__(parent)
        self.setObjectName("OverviewGuidedCellPicker")
        self.setMinimumSize(240, 300)
        self.setToolTip(tr("Click a cell, or drag over several, to choose where this well's images go on the page."))
        self.rows, self.columns, self.aspect = 3, 2, 595.2756 / 841.8898
        self.occupied = {}
        self.selection = (0, 0, 1, 1)
        self._anchor = None

    def configure(self, rows, columns, aspect, occupied):
        """Set the grid, the page's width over height and the labelled taken cells."""
        self.rows, self.columns, self.aspect = max(1, rows), max(1, columns), aspect
        self.occupied = dict(occupied)
        row, column, row_span, column_span = self.selection
        row, column = min(row, self.rows - 1), min(column, self.columns - 1)
        self.selection = (row, column, min(row_span, self.rows - row), min(column_span, self.columns - column))
        self.update()

    def _paper(self):
        """The page rectangle inside the widget."""
        from PySide6.QtCore import QRectF

        width, height = self.width() - 16, self.height() - 16
        if width / max(height, 1) > self.aspect:
            width = height * self.aspect
        else:
            height = width / self.aspect
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def _cell_at(self, point):
        """The ``(row, column)`` under ``point``, kept inside the grid."""
        paper = self._paper()
        column = int((point.x() - paper.x()) / max(paper.width(), 1) * self.columns)
        row = int((point.y() - paper.y()) / max(paper.height(), 1) * self.rows)
        return min(max(row, 0), self.rows - 1), min(max(column, 0), self.columns - 1)

    def mousePressEvent(self, event):
        """Start choosing cells."""
        self._anchor = self._cell_at(event.position())
        self._choose(self._anchor)

    def mouseMoveEvent(self, event):
        """Grow the chosen block of cells."""
        if self._anchor is not None:
            self._choose(self._cell_at(event.position()))

    def mouseReleaseEvent(self, event):
        """Finish choosing."""
        self._anchor = None

    def _choose(self, cell):
        """Choose the block from the anchor to ``cell``."""
        top, bottom = sorted((self._anchor[0], cell[0]))
        left, right = sorted((self._anchor[1], cell[1]))
        self.selection = (top, left, bottom - top + 1, right - left + 1)
        self.update()
        self.changed.emit()

    def paintEvent(self, event):
        """Draw the paper, its grid, the taken cells and the chosen block."""
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QColor, QPainter, QPen
        from ..iconset import active_theme
        from ..theme import palette_for

        palette = palette_for(active_theme())
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        paper = self._paper()
        painter.fillRect(paper.translated(2, 2), QColor(0, 0, 0, 50))
        painter.fillRect(paper, QColor("#ffffff"))
        cell_width, cell_height = paper.width() / self.columns, paper.height() / self.rows
        for (row, column), label in self.occupied.items():
            rect = QRectF(paper.x() + column * cell_width, paper.y() + row * cell_height, cell_width, cell_height)
            painter.fillRect(rect.adjusted(2, 2, -2, -2), QColor("#d4ece9"))
            painter.setPen(QColor("#0a6f6a"))
            painter.drawText(rect.adjusted(4, 4, -4, -4), Qt.AlignCenter | Qt.TextWordWrap, label)
        painter.setPen(QPen(QColor("#c9d5e3"), 1))
        for column in range(1, self.columns):
            painter.drawLine(paper.x() + column * cell_width, paper.y(), paper.x() + column * cell_width, paper.bottom())
        for row in range(1, self.rows):
            painter.drawLine(paper.x(), paper.y() + row * cell_height, paper.right(), paper.y() + row * cell_height)
        row, column, row_span, column_span = self.selection
        chosen = QRectF(paper.x() + column * cell_width, paper.y() + row * cell_height,
                        column_span * cell_width, row_span * cell_height).adjusted(2, 2, -2, -2)
        accent = QColor(palette["accent"])
        fill = QColor(accent)
        fill.setAlpha(60)
        painter.fillRect(chosen, fill)
        painter.setPen(QPen(accent, 2))
        painter.drawRect(chosen)
        painter.end()


class _OverviewGuidedExport(QDialog):
    """Export chosen wells to PDF step by step: wells, fields, placement and captions, preview.

    The result is a :class:`spacr.qt.widgets.picture_export._CanvasLayout`,
    the same model the Custom export edits, so the preview can be refined
    there before it is written.

    :param browser: the Overview plate browser the wells come from.
    """

    def __init__(self, browser):
        """Build the four steps, starting from the wells selected in the plate."""
        from PySide6.QtWidgets import QStackedWidget

        super().__init__(browser)
        from .glass import NO_GLASS

        self.setWindowFlags(Qt.WindowType.Widget)
        self.setProperty(NO_GLASS, True)
        self.setObjectName("OverviewGuidedExport")
        self.setWindowTitle(tr("Guided export"))
        self._browser = browser
        self._source = browser._canvas_snapshot()
        self._wells = []
        self._fields = {}
        self._placements = []
        self._index = 0
        self._frame = None
        self._page = 0
        self._composer = None
        self._jobs = JobRunner(self, app_key="field_overview", user_visible=False)
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        self.heading = QLabel("")
        self.heading.setObjectName("OverviewGuidedHeading")
        font = self.heading.font()
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() * 1.15)
        self.heading.setFont(font)
        self.hint = QLabel("")
        self.hint.setWordWrap(True)
        root.addWidget(self.heading)
        root.addWidget(self.hint)
        self.steps = QStackedWidget()
        self.steps.addWidget(self._build_wells_step())
        self.steps.addWidget(self._build_fields_step())
        self.steps.addWidget(self._build_place_step())
        self.preview_page = QWidget()
        QVBoxLayout(self.preview_page).setContentsMargins(0, 0, 0, 0)
        self.steps.addWidget(self.preview_page)
        root.addWidget(self.steps, 1)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        buttons = QHBoxLayout()
        self.custom_button = QPushButton(tr("Open in Custom export"))
        self.custom_button.setToolTip(tr("Continue in the Custom export canvas with the wells placed so far."))
        self.custom_button.clicked.connect(self.open_custom)
        self.back_button = QPushButton(tr("Back"))
        self.back_button.clicked.connect(self.back)
        self.next_button = QPushButton(tr("Next"))
        self.next_button.setObjectName("OverviewGuidedNext")
        self.next_button.clicked.connect(self.next)
        self.export_button = QPushButton(tr("Export PDF…"))
        self.export_button.setToolTip(tr("Write the previewed pages to one PDF."))
        self.export_button.clicked.connect(self._ask_export)
        buttons.addWidget(self.custom_button)
        buttons.addStretch(1)
        for button in (self.back_button, self.next_button, self.export_button):
            buttons.addWidget(button)
        root.addLayout(buttons)
        self.resize(1100, 780)
        from ..theme import ensure_widget_qss_applied, register_widget_qss

        register_widget_qss("OverviewGuidedExport", _overview_guided_qss, replace=True)
        ensure_widget_qss_applied("OverviewGuidedExport", root=self)
        self._prefill()
        self._show_step(0)

    def _build_wells_step(self):
        """Step 1: plates, then wells by condition, by multi-select or one well."""
        from PySide6.QtWidgets import QButtonGroup, QGridLayout, QListWidget, QRadioButton

        page = QWidget()
        grid = QGridLayout(page)
        self.plates = QListWidget()
        self.plates.setObjectName("OverviewGuidedPlates")
        self.plates.setToolTip(tr("Plates the wells are chosen from."))
        self.plates.setFixedWidth(200)
        for plate in dict.fromkeys(field.plate for field in self._browser._fields):
            item = self._checkable(self.plates, plate, plate)
            item.setCheckState(Qt.Checked)
        self.plates.itemChanged.connect(self._plates_changed)
        self.mode_group = QButtonGroup(self)
        modes = QHBoxLayout()
        modes.addWidget(QLabel(tr("Choose wells")))
        for index, (caption, tooltip) in enumerate(_OVERVIEW_GUIDED_WELL_MODES):
            button = QRadioButton(tr(caption))
            button.setToolTip(tr(tooltip))
            self.mode_group.addButton(button, index)
            modes.addWidget(button)
        modes.addStretch(1)
        self.mode_group.idToggled.connect(self._mode_changed)
        self.conditions = QListWidget()
        self.conditions.setObjectName("OverviewGuidedConditions")
        self.conditions.setToolTip(tr("Annotation values and comparison groups; tick one or more."))
        self.conditions.itemChanged.connect(self._conditions_changed)
        self.well_list = QListWidget()
        self.well_list.setObjectName("OverviewGuidedWells")
        self.well_list.setToolTip(tr("Wells to export, in this order."))
        self.well_list.itemChanged.connect(self._conditions_changed)
        self.single = QComboBox()
        self.single.setToolTip(tr("The one well to export."))
        self.single.currentIndexChanged.connect(self._conditions_changed)
        self.well_count = QLabel("")
        self.well_count.setObjectName("OverviewGuidedCount")
        grid.addLayout(modes, 0, 0, 1, 3)
        grid.addWidget(QLabel(tr("Plates")), 1, 0)
        grid.addWidget(QLabel(tr("Conditions")), 1, 1)
        grid.addWidget(QLabel(tr("Wells")), 1, 2)
        grid.addWidget(self.plates, 2, 0, 2, 1)
        grid.addWidget(self.conditions, 2, 1, 2, 1)
        wells = QVBoxLayout()
        wells.addWidget(self.single)
        wells.addWidget(self.well_list, 1)
        grid.addLayout(wells, 2, 2, 2, 1)
        grid.addWidget(self.well_count, 4, 0, 1, 3)
        grid.setRowStretch(3, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        return page

    def _build_fields_step(self):
        """Step 2: every field, the first N, or picked fields per well."""
        from PySide6.QtWidgets import QButtonGroup, QRadioButton, QSpinBox, QTreeWidget

        page = QWidget()
        column = QVBoxLayout(page)
        self.field_mode = QButtonGroup(self)
        row = QHBoxLayout()
        for index, (caption, tooltip) in enumerate(_OVERVIEW_GUIDED_FIELD_MODES):
            button = QRadioButton(tr(caption))
            button.setToolTip(tr(tooltip))
            self.field_mode.addButton(button, index)
            row.addWidget(button)
            if index == 1:
                self.first_count = QSpinBox()
                self.first_count.setRange(1, 99)
                self.first_count.setValue(1)
                self.first_count.setToolTip(tr("How many fields of each well First fields takes."))
                self.first_count.valueChanged.connect(self._field_mode_changed)
                row.addWidget(self.first_count)
        row.addStretch(1)
        self.field_mode.button(0).setChecked(True)
        self.field_mode.idToggled.connect(self._field_mode_changed)
        self.field_tree = QTreeWidget()
        self.field_tree.setObjectName("OverviewGuidedFields")
        self.field_tree.setHeaderHidden(True)
        self.field_tree.setToolTip(tr("Fields of each chosen well; the selected ones are exported. Drag over fields "
                                      "to select them, Shift-click for a range, Ctrl-click to add or remove one, "
                                      "Ctrl+A for every field, or right-click to select and deselect."))
        self.field_tree.setSelectionMode(QTreeWidget.ExtendedSelection)
        self.field_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.field_tree.customContextMenuRequested.connect(self._field_menu)
        self.field_tree.itemSelectionChanged.connect(self._field_selection_changed)
        self.field_tree.itemChanged.connect(self._field_ticked)
        self._field_syncing = False
        self._field_menu_shown = None
        from .sortable_table import install_sorting

        install_sorting(self.field_tree)
        column.addLayout(row)
        column.addWidget(self.field_tree, 1)
        return page

    def _build_place_step(self):
        """Step 3: where the well goes on the page and what is printed around it."""
        from PySide6.QtWidgets import QGridLayout, QLineEdit, QListWidget, QSpinBox

        page = QWidget()
        outer = QHBoxLayout(page)
        left = QVBoxLayout()
        sizes = QHBoxLayout()
        self.page_size = QComboBox()
        for name in ("A4", "Letter"):
            self.page_size.addItem(tr(name), name)
        self.page_size.setToolTip(tr("Paper size of every page."))
        self.orientation = QComboBox()
        self.orientation.addItem(tr("Portrait"), "portrait")
        self.orientation.addItem(tr("Landscape"), "landscape")
        self.orientation.setToolTip(tr("Page orientation of every page."))
        self.grid_rows, self.grid_columns = QSpinBox(), QSpinBox()
        for spin, value, tooltip in ((self.grid_rows, 3, "Rows of the page grid."),
                                     (self.grid_columns, 2, "Columns of the page grid.")):
            spin.setRange(1, 12)
            spin.setValue(value)
            spin.setToolTip(tr(tooltip))
        for widget in (self.page_size, self.orientation, QLabel(tr("Rows")), self.grid_rows,
                       QLabel(tr("Columns")), self.grid_columns):
            sizes.addWidget(widget)
        for control in (self.page_size, self.orientation):
            control.currentIndexChanged.connect(self._sync_picker)
        for control in (self.grid_rows, self.grid_columns):
            control.valueChanged.connect(self._sync_picker)
        self.picker = _OverviewCellPicker()
        paging = QHBoxLayout()
        self.page_label = QLabel("")
        earlier, later = QPushButton(tr("Previous page")), QPushButton(tr("Next page"))
        earlier.clicked.connect(functools.partial(self._turn_page, -1))
        later.clicked.connect(functools.partial(self._turn_page, 1))
        for widget in (earlier, self.page_label, later):
            paging.addWidget(widget)
        left.addLayout(sizes)
        left.addWidget(self.picker, 1)
        left.addLayout(paging)
        right = QGridLayout()
        right.addWidget(QLabel(tr("Print")), 0, 0)
        right.addWidget(QLabel(tr("Where")), 0, 1)
        self.sides = {}
        for row, item in enumerate(_OVERVIEW_GUIDED_ITEMS, start=1):
            combo = QComboBox()
            combo.setObjectName("OverviewGuidedSide_" + item)
            for side in _OVERVIEW_GUIDED_SIDES:
                combo.addItem(tr(_OVERVIEW_GUIDED_SIDE_CAPTIONS[side]), side)
            combo.setCurrentIndex(combo.findData(_OVERVIEW_GUIDED_DEFAULT_SIDES.get(item, "none")))
            combo.setToolTip(tr("Where {item} is printed around this well's images.",
                                item=tr(_OVERVIEW_GUIDED_CAPTIONS[item]).lower()))
            self.sides[item] = combo
            right.addWidget(QLabel(tr(_OVERVIEW_GUIDED_CAPTIONS[item])), row, 0)
            right.addWidget(combo, row, 1)
        self.free_text = QLineEdit()
        self.free_text.setPlaceholderText(tr("Free text for this well"))
        self.free_text.setToolTip(tr("Printed where Free text is placed."))
        right.addWidget(self.free_text, len(_OVERVIEW_GUIDED_ITEMS) + 1, 0, 1, 2)
        self.measurement_list = QListWidget()
        self.measurement_list.setObjectName("OverviewGuidedMeasurements")
        self.measurement_list.setMaximumHeight(110)
        self.measurement_list.setToolTip(tr("Image-level measurements to print, averaged over the well's chosen fields."))
        right.addWidget(QLabel(tr("Measurements")), len(_OVERVIEW_GUIDED_ITEMS) + 2, 0, 1, 2)
        right.addWidget(self.measurement_list, len(_OVERVIEW_GUIDED_ITEMS) + 3, 0, 1, 2)
        self.apply_rest = Toggle(tr("Apply this layout to the remaining wells"))
        self.apply_rest.setToolTip(tr("Place every remaining well the same way, in the next free cells, "
                                      "with the same information around it."))
        self.display_note = Toggle(tr("Add a page with the display settings"))
        self.display_note.setToolTip(tr("End the PDF with a page listing channels, normalization and colour mode."))
        self.scale_bars = Toggle(tr("Scale bars"))
        known = bool(self._browser._overview_options.get("pixel_size_um"))
        self.scale_bars.setEnabled(known)
        self.scale_bars.setToolTip(tr("Draw a scale bar on every image.") if known else
                                   tr("The pixel size of these images is not known, so no scale bar can be drawn."))
        right.addWidget(self.apply_rest, len(_OVERVIEW_GUIDED_ITEMS) + 4, 0, 1, 2)
        right.addWidget(self.scale_bars, len(_OVERVIEW_GUIDED_ITEMS) + 5, 0, 1, 2)
        right.addWidget(self.display_note, len(_OVERVIEW_GUIDED_ITEMS) + 6, 0, 1, 2)
        right.setRowStretch(len(_OVERVIEW_GUIDED_ITEMS) + 7, 1)
        outer.addLayout(left, 3)
        outer.addLayout(right, 2)
        return page

    @staticmethod
    def _checkable(widget, text, data, checked=False):
        """Add a checkable item carrying ``data`` to a list widget."""
        from PySide6.QtWidgets import QListWidgetItem

        item = QListWidgetItem(text)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        item.setData(Qt.UserRole, data)
        widget.addItem(item)
        return item

    def _prefill(self):
        """Offer conditions and wells; start from the plate's selected wells, if any."""
        selected = set(self._browser._selected_wells)
        self.mode_group.button(1 if selected else 0).setChecked(True)
        self._fill_wells(selected)
        self._fill_conditions()
        self._mode_changed()

    def _chosen_plates(self):
        """The ticked plates."""
        return [self.plates.item(index).data(Qt.UserRole) for index in range(self.plates.count())
                if self.plates.item(index).checkState() == Qt.Checked]

    def _all_wells(self):
        """Every ``(plate, well)`` of the ticked plates, in plate then reading order."""
        plates = self._chosen_plates()
        wells = dict.fromkeys((field.plate, field.well) for field in self._browser._fields
                              if field.plate in plates and field.well and not field.unresolved)
        return sorted(wells, key=lambda address: (plates.index(address[0]), self._browser._natural(address[1])))

    def _well_caption(self, address):
        """``plate / well`` and its annotations, for the well lists."""
        annotations = self._browser._annotations.get(address) or {}
        text = address[0] + " / " + address[1]
        if annotations:
            text += " — " + ", ".join(f"{key}: {value}" for key, value in sorted(annotations.items()))
        return text

    def _fill_wells(self, checked=()):
        """List the wells of the ticked plates, ticking ``checked``."""
        self.well_list.blockSignals(True)
        self.well_list.clear()
        self.single.clear()
        for address in self._all_wells():
            self._checkable(self.well_list, self._well_caption(address), address, address in checked)
            self.single.addItem(self._well_caption(address), address)
        self.well_list.blockSignals(False)

    def _condition_options(self):
        """Every annotation value and comparison group, with the wells each matches."""
        wells = self._all_wells()
        options = {}
        for address in wells:
            for key, value in sorted((self._browser._annotations.get(address) or {}).items()):
                options.setdefault(f"{key}: {value}", []).append(address)
        builder = self._browser.group_builder
        for definition, name in zip(builder.definitions(), builder.names()):
            label = tr("Group") + ": " + (name or _overview_group_label(definition))
            options[label] = [address for address in wells if all(
                str((self._browser._annotations.get(address) or {}).get(key)) == str(value)
                for key, value in definition.items())]
        return options

    def _fill_conditions(self):
        """List the conditions of the ticked plates."""
        self.conditions.blockSignals(True)
        self.conditions.clear()
        for label, wells in self._condition_options().items():
            self._checkable(self.conditions, tr("{condition} ({count} wells)", condition=label, count=len(wells)),
                            label)
        self.conditions.blockSignals(False)
        if not self.conditions.count():
            self.conditions.addItem(tr("Annotate wells, or define comparison groups, to choose by condition."))

    def _plates_changed(self, *_):
        """Refill the conditions and wells for the ticked plates."""
        self._fill_wells(set(self.chosen_wells()))
        self._fill_conditions()

    def _mode_changed(self, *_):
        """Enable the controls of the chosen way of picking wells."""
        mode = self.mode_group.checkedId()
        self.conditions.setEnabled(mode == 0)
        self.well_list.setEnabled(mode == 1)
        self.single.setEnabled(mode == 2)
        self.single.setVisible(mode == 2)
        self._update_count()

    def _conditions_changed(self, *_):
        """Count the wells the ticked conditions match."""
        self._update_count()

    def _update_count(self):
        """Say how many wells are chosen."""
        self.well_count.setText(tr("{count} wells chosen", count=len(self.chosen_wells())))

    def chosen_wells(self):
        """The wells step 1 chooses, in plate then reading order."""
        mode = self.mode_group.checkedId()
        if mode == 2:
            address = self.single.currentData()
            return [tuple(address)] if address is not None else []
        if mode == 1:
            return [tuple(self.well_list.item(index).data(Qt.UserRole)) for index in range(self.well_list.count())
                    if self.well_list.item(index).checkState() == Qt.Checked]
        options = self._condition_options()
        ticked = [self.conditions.item(index).data(Qt.UserRole) for index in range(self.conditions.count())
                  if self.conditions.item(index).checkState() == Qt.Checked]
        matched = {address for label in ticked for address in options.get(label, ())}
        return [address for address in self._all_wells() if address in matched]

    def choose_wells(self, mode, values=()):
        """Choose wells as step 1 does: ``"condition"``, ``"wells"`` or ``"one"``.

        :param values: condition labels, ``(plate, well)`` pairs, or one pair.
        """
        index = {"condition": 0, "wells": 1, "one": 2}[mode]
        self.mode_group.button(index).setChecked(True)
        wanted = list(values) if mode != "one" else [tuple(values)]
        if mode == "condition":
            for row in range(self.conditions.count()):
                item = self.conditions.item(row)
                if item.flags() & Qt.ItemIsUserCheckable:
                    item.setCheckState(Qt.Checked if item.data(Qt.UserRole) in wanted else Qt.Unchecked)
        elif mode == "wells":
            for row in range(self.well_list.count()):
                item = self.well_list.item(row)
                item.setCheckState(Qt.Checked if tuple(item.data(Qt.UserRole)) in {tuple(w) for w in wanted}
                                   else Qt.Unchecked)
        else:
            matches = [index for index in range(self.single.count()) if tuple(self.single.itemData(index)) == wanted[0]]
            self.single.setCurrentIndex(matches[0] if matches else 0)
        self._update_count()
        return self.chosen_wells()

    def _well_fields(self, address):
        """The fields of one well at the shown time and Z, in field order."""
        return self._browser._canvas_well_fields(*address)

    def _fill_fields(self):
        """List each chosen well's fields, ticking the ones the field mode takes."""
        from .sortable_table import SortableTreeItem

        self._field_syncing = True
        self.field_tree.clear()
        mode, count = self.field_mode.checkedId(), self.first_count.value()
        for address in self._wells:
            parent = SortableTreeItem([address[0] + " / " + address[1]])
            parent.setData(0, Qt.UserRole, list(address))
            parent.setFlags(parent.flags() & ~Qt.ItemIsSelectable)
            self.field_tree.addTopLevelItem(parent)
            for index, field in enumerate(self._well_fields(address)):
                child = SortableTreeItem([tr("Field {field}", field=field.field)])
                child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                child.setData(0, Qt.UserRole, _overview_key_text(_overview_image_key(field)))
                child.setCheckState(0, Qt.Checked if mode == 0 or (mode == 1 and index < count) or mode == 2
                                    and index == 0 else Qt.Unchecked)
                parent.addChild(child)
            parent.setExpanded(True)
        self._field_syncing = False
        self._select_ticked()

    def _field_children(self):
        """Every field row of the field tree, well by well."""
        return [parent.child(index) for row in range(self.field_tree.topLevelItemCount())
                for parent in (self.field_tree.topLevelItem(row),) for index in range(parent.childCount())]

    def _select_ticked(self):
        """Select exactly the ticked field rows."""
        self._field_syncing = True
        try:
            for child in self._field_children():
                child.setSelected(child.checkState(0) == Qt.Checked)
        finally:
            self._field_syncing = False

    def _to_pick_mode(self):
        """Switch to Pick fields without retiking, once the user picks by hand."""
        if self.field_mode.checkedId() != 2:
            self.field_mode.blockSignals(True)
            self.field_mode.button(2).setChecked(True)
            self.field_mode.blockSignals(False)

    def _field_selection_changed(self):
        """Tick the selected fields and untick the others: selecting is picking."""
        if self._field_syncing:
            return
        self._field_syncing = True
        try:
            for child in self._field_children():
                child.setCheckState(0, Qt.Checked if child.isSelected() else Qt.Unchecked)
        finally:
            self._field_syncing = False
        self._to_pick_mode()

    def _field_ticked(self, item, _column=0):
        """A ticked or unticked field is selected or deselected with it."""
        if self._field_syncing or item.parent() is None:
            return
        self._field_syncing = True
        try:
            item.setSelected(item.checkState(0) == Qt.Checked)
        finally:
            self._field_syncing = False
        self._to_pick_mode()

    def pick_fields(self, items, chosen):
        """Select (``chosen``) or deselect field rows; a well row stands for all of its fields.

        :returns: the field rows changed.
        """
        rows = []
        for item in items:
            rows.extend([item.child(index) for index in range(item.childCount())] if item.parent() is None else [item])
        self.field_tree.blockSignals(True)
        try:
            for row in rows:
                row.setSelected(bool(chosen))
        finally:
            self.field_tree.blockSignals(False)
        self._field_selection_changed()
        self.field_tree.viewport().update()
        return rows

    def _field_menu(self, position):
        """Right-click menu of the field tree: select or deselect, here or everywhere; returns it."""
        from PySide6.QtWidgets import QMenu

        item = self.field_tree.itemAt(position)
        if item is not None and item.parent() is not None and item.isSelected():
            targets = [child for child in self._field_children() if child.isSelected()]
        else:
            targets = [item] if item is not None else []
        menu = QMenu(self.field_tree)
        menu.setObjectName("OverviewGuidedFieldMenu")
        for caption, items, chosen in (("Select", targets, True), ("Deselect", targets, False),
                                       ("Select all fields", self._field_children(), True),
                                       ("Deselect all fields", self._field_children(), False)):
            action = menu.addAction(tr(caption))
            action.setEnabled(bool(items))
            action.triggered.connect(functools.partial(self._menu_pick, items, chosen))
        menu.popup(self.field_tree.viewport().mapToGlobal(position))
        self._field_menu_shown = menu
        return menu

    def _menu_pick(self, items, chosen, *_):
        """Run a Select or Deselect entry of the field menu."""
        self.pick_fields(items, chosen)

    def _field_mode_changed(self, *_):
        """Retick the fields for the chosen mode."""
        self._fill_fields()

    def _tree_wells(self):
        """``[(address, {image key text: field}, [child items])]`` for each well in the field tree."""
        wells = []
        for row in range(self.field_tree.topLevelItemCount()):
            parent = self.field_tree.topLevelItem(row)
            address = tuple(parent.data(0, Qt.UserRole))
            fields = {_overview_key_text(_overview_image_key(field)): field for field in self._well_fields(address)}
            wells.append((address, fields, [parent.child(index) for index in range(parent.childCount())]))
        return wells

    def set_fields(self, mode, count=1, picks=None):
        """Choose fields as step 2 does: ``"all"``, ``"first"`` (``count``) or ``"pick"``.

        :param picks: for ``"pick"``, ``{(plate, well): [field names]}``.
        """
        self.first_count.setValue(int(count))
        self.field_mode.button({"all": 0, "first": 1, "pick": 2}[mode]).setChecked(True)
        self._fill_fields()
        if mode == "pick":
            for address, fields, children in self._tree_wells():
                wanted = {str(name) for name in (picks or {}).get(address, ())}
                for child in children:
                    field = fields[child.data(0, Qt.UserRole)]
                    self._field_syncing = True
                    child.setCheckState(0, Qt.Checked if str(field.field) in wanted else Qt.Unchecked)
                    self._field_syncing = False
            self._select_ticked()

    def _chosen_fields(self):
        """``{(plate, well): [fields]}`` from the field tree, in the order of the chosen wells."""
        chosen = {}
        for address, fields, children in self._tree_wells():
            chosen[address] = [fields[child.data(0, Qt.UserRole)] for child in children
                               if child.checkState(0) == Qt.Checked]
        return {address: chosen.get(address, []) for address in self._wells}

    def _show_step(self, step):
        """Show step ``step`` with its heading, hint and buttons."""
        self.steps.setCurrentIndex(step)
        if step == 0:
            self.heading.setText(tr("Step 1 of 4: choose wells"))
            self.hint.setText(tr("Choose wells from one or several plates: by condition, by ticking wells, or one well."))
        elif step == 1:
            self.heading.setText(tr("Step 2 of 4: choose fields"))
            self.hint.setText(tr("Export every field, the first fields, or pick fields in each well."))
        elif step == 2:
            address = self._wells[self._index]
            self.heading.setText(tr("Step 3 of 4: well {number} of {total}, {well}", number=self._index + 1,
                                    total=len(self._wells), well=address[0] + " / " + address[1]))
            self.hint.setText(tr("Choose where this well's images go on the page and what is printed around them."))
        else:
            self.heading.setText(tr("Step 4 of 4: preview"))
            self.hint.setText(tr("Check the pages, adjust any box, choose the DPI under Settings…, then export the PDF."))
        self.back_button.setEnabled(step > 0)
        self.next_button.setVisible(step < 3)
        self.export_button.setVisible(step == 3)
        self.next_button.setText(tr("Next well") if step == 2 and self._index + 1 < len(self._wells)
                                 else tr("Preview") if step == 2 else tr("Next"))

    def next(self):
        """Go on: from the wells to the fields, from well to well, then to the preview."""
        step = self.steps.currentIndex()
        if step == 0:
            self._wells = self.chosen_wells()
            if not self._wells:
                self.status.setText(tr("Choose at least one well."))
                return False
            self._fill_fields()
            self.status.setText("")
            self._show_step(1)
            return True
        if step == 1:
            self._fields = self._chosen_fields()
            if not any(self._fields.values()):
                self.status.setText(tr("Choose at least one field."))
                return False
            self._placements = []
            self._index = 0
            self._page = 0
            self._load_measurements()
            self.status.setText("")
            self._enter_well()
            return True
        if step == 2:
            self._record_current()
            if self.apply_rest.isChecked():
                self._place_remaining()
            if self._index + 1 < len(self._wells) and not self.apply_rest.isChecked():
                self._index += 1
                self._enter_well()
                return True
            self._show_preview()
            return True
        return False

    def back(self):
        """Go back one well, or one step."""
        step = self.steps.currentIndex()
        if step == 3:
            self._index = len(self._placements) - 1
            self._placements = self._placements[:-1]
            self._enter_well()
        elif step == 2 and self._index > 0:
            self._index -= 1
            self._placements = self._placements[:self._index]
            self._enter_well()
        elif step > 0:
            self._show_step(step - 1)

    def _occupied(self):
        """``{page: {(row, column): label}}`` of the wells placed so far."""
        taken = {}
        for placement in self._placements:
            row, column, row_span, column_span = placement["cell"]
            for r in range(row, row + row_span):
                for c in range(column, column + column_span):
                    taken.setdefault(placement["page"], {})[(r, c)] = placement["well"][1]
        return taken

    def _page_aspect(self):
        """The page's width over height."""
        from .picture_export import _CANVAS_PAGE_SIZES

        width, height = _CANVAS_PAGE_SIZES[self.page_size.currentData()]
        return height / width if self.orientation.currentData() == "landscape" else width / height

    def _sync_picker(self, *_):
        """Show the current page's grid and taken cells."""
        self.picker.configure(self.grid_rows.value(), self.grid_columns.value(), self._page_aspect(),
                              self._occupied().get(self._page, {}))
        self.page_label.setText(tr("Page {number}", number=self._page + 1))

    def _turn_page(self, step):
        """Place the current well on another page."""
        self._page = max(0, self._page + int(step))
        self._sync_picker()

    def _enter_well(self):
        """Show the placement step for the current well, in the next free cell."""
        previous = self._placements[-1] if self._placements else None
        span = previous["cell"][2:] if previous else (1, 1)
        taken = {page: set(cells) for page, cells in self._occupied().items()}
        self._page, cell = _overview_free_slot(taken, self.grid_rows.value(), self.grid_columns.value(), span,
                                               previous["page"] if previous else 0)
        self.picker.selection = cell
        self.free_text.clear()
        self._sync_picker()
        self._show_step(2)

    def place(self, cell=None, sides=None, text=None, measurements=None, apply_rest=None, page=None):
        """Set the current well's placement as step 3 does; any argument left out keeps its value.

        :param cell: ``(row, column, row span, column span)`` on the page grid.
        :param sides: ``{item: side}`` for the items in ``_OVERVIEW_GUIDED_ITEMS``.
        :param text: the free text.
        :param measurements: measurement columns to print.
        :param apply_rest: place the remaining wells the same way on Next.
        :param page: the page, from 0.
        """
        if page is not None:
            self._page = int(page)
        if cell is not None:
            self.picker.selection = tuple(cell)
        for item, side in (sides or {}).items():
            self.sides[item].setCurrentIndex(self.sides[item].findData(side))
        if text is not None:
            self.free_text.setText(text)
        if measurements is not None:
            for row in range(self.measurement_list.count()):
                item = self.measurement_list.item(row)
                item.setCheckState(Qt.Checked if item.data(Qt.UserRole) in measurements else Qt.Unchecked)
        if apply_rest is not None:
            self.apply_rest.setChecked(bool(apply_rest))
        self._sync_picker()

    def _current_sides(self):
        """``{item: side}`` of the information printed around the current well."""
        return {item: combo.currentData() for item, combo in self.sides.items() if combo.currentData() != "none"}

    def _chosen_measurements(self):
        """The ticked measurement columns."""
        return [self.measurement_list.item(row).data(Qt.UserRole) for row in range(self.measurement_list.count())
                if self.measurement_list.item(row).checkState() == Qt.Checked]

    def _record_current(self):
        """Keep the current well's placement and captions."""
        self._placements = self._placements[:self._index]
        self._placements.append({"well": self._wells[self._index], "page": self._page,
                                 "cell": tuple(self.picker.selection), "sides": self._current_sides(),
                                 "text": self.free_text.text(), "measurements": self._chosen_measurements()})

    def _place_remaining(self):
        """Place every remaining well like the current one, in the next free cells."""
        current = self._placements[-1]
        while len(self._placements) < len(self._wells):
            taken = {page: set(cells) for page, cells in self._occupied().items()}
            page, cell = _overview_free_slot(taken, self.grid_rows.value(), self.grid_columns.value(),
                                             current["cell"][2:], current["page"])
            self._placements.append(dict(current, well=self._wells[len(self._placements)], page=page, cell=cell))
        self._index = len(self._wells) - 1

    def _load_measurements(self):
        """Read the image-level measurements off the GUI thread, if they were exported."""
        self.measurement_list.clear()
        self._frame = None
        source = self._browser._metadata_source
        self.sides["measurements"].setEnabled(False)
        self.sides["measurements"].setToolTip(tr("Export image-level measurements first to print them here."))
        if source:
            self._jobs.submit(functools.partial(_overview_guided_frame, source), self._measurements_loaded)

    def _measurements_loaded(self, frame):
        """Offer the numeric measurement columns."""
        if frame is None:
            return
        self._frame = frame
        reserved = {"plate", "plateID", "well", "wellID", "field", "fieldID", "row", "column", "time_id", "z_id"}
        for name in frame.select_dtypes("number").columns:
            if name not in reserved:
                self._checkable(self.measurement_list, str(name), str(name))
        self.sides["measurements"].setEnabled(True)
        self.sides["measurements"].setToolTip(tr("Where measurements are printed around this well's images."))

    def _measurement_text(self, address, fields, columns):
        """``name: mean`` over the well's chosen fields for each measurement column."""
        frame = self._frame
        if frame is None or not columns:
            return ""
        plate_column = "plateID" if "plateID" in frame.columns else "plate"
        well_column = "wellID" if "wellID" in frame.columns else "well"
        plate_ids = {str(field.plate_id or field.plate) for field in fields} | {address[0]}
        names = {str(field.field_id or field.field) for field in fields}
        rows = frame[frame[plate_column].astype(str).isin(plate_ids) & (frame[well_column].astype(str) == address[1])]
        if "fieldID" in rows.columns:
            rows = rows[rows["fieldID"].astype(str).isin(names)]
        lines = []
        for column in columns:
            values = rows[column].dropna() if column in rows.columns else ()
            lines.append(f"{column}: {float(values.mean()):.4g}" if len(values) else f"{column}: –")
        return "\n".join(lines)

    def _caption_texts(self, placement):
        """``{side: text}`` printed around one placed well."""
        address = placement["well"]
        fields = self._fields.get(address, [])
        annotations = self._browser._annotations.get(address) or {}
        condition = ", ".join(f"{key}: {value}" for key, value in sorted(annotations.items()))
        channels = ", ".join(self._browser._canvas_channel_names())
        values = {"plate": tr("Plate {plate}", plate=address[0]), "well": tr("Well {well}", well=address[1]),
                  "condition": condition,
                  "field": tr("Fields {fields}", fields=", ".join(str(field.field) for field in fields)),
                  "channels": tr("Channels: {channels}", channels=channels),
                  "measurements": self._measurement_text(address, fields, placement.get("measurements") or ()),
                  "text": placement.get("text") or ""}
        texts = {}
        for item in _OVERVIEW_GUIDED_ITEMS:
            side = placement["sides"].get(item)
            if side and values[item]:
                separator = "\n" if item == "measurements" or side in ("left", "right") else " · "
                texts[side] = texts[side] + separator + values[item] if side in texts else values[item]
        return texts

    def canvas_layout(self):
        """The canvas layout of the wells placed so far."""
        pixel_size = self._browser._overview_options.get("pixel_size_um")
        bars = bool(pixel_size) and self.scale_bars.isChecked()
        entries = [{"page": placement["page"], "cell": placement["cell"],
                    "images": [dict(_overview_canvas_image(field, pixel_size), scale_bar=bars)
                               for field in self._fields.get(placement["well"], [])],
                    "captions": self._caption_texts(placement)} for placement in self._placements]
        note = self._browser._canvas_display_note() if self.display_note.isChecked() else ()
        return _overview_guided_layout(entries, self.page_size.currentData(), self.orientation.currentData(),
                                       self.grid_rows.value(), self.grid_columns.value(), note)

    def _show_preview(self):
        """Show the pages in an embedded canvas where boxes can still be adjusted."""
        from .picture_export import _CanvasComposer

        holder = self.preview_page.layout()
        if self._composer is not None:
            self._composer.close()
            self._composer.deleteLater()
        self._composer = _CanvasComposer(self.canvas_layout(), _OverviewCanvasSource(self._browser),
                                         self._browser._metadata_source or "", self.preview_page)
        self._composer.setWindowFlags(Qt.Widget)
        self._composer.export_button.hide()
        holder.addWidget(self._composer)
        self._composer.show()
        self._show_step(3)

    @property
    def composer(self):
        """The preview's canvas, once the preview is shown."""
        return self._composer

    def open_custom(self):
        """Hand the wells placed so far to the Custom export canvas; returns the canvas."""
        if self.steps.currentIndex() == 2:
            self._record_current()
        layout = (self._composer.canvas if self._composer is not None and self.steps.currentIndex() == 3
                  else self.canvas_layout())
        from .picture_export import _CanvasLayout

        composer = self._browser._open_custom_export(_CanvasLayout.from_dict(layout.to_dict()))
        self.close()
        return composer

    def _ask_export(self):
        """Write the preview through the canvas, at the DPI chosen in its Settings…."""
        if self._composer is not None:
            self._composer._ask_export()

    def export_pdf(self, path):
        """Write the preview to ``path`` at the DPI chosen in the canvas's Settings…; returns whether it started."""
        if self._composer is None:
            return False
        return self._composer.export_pdf(path)

    def closeEvent(self, event):
        """Stop the measurement read with the window."""
        self._jobs.cancel()
        if self._composer is not None:
            self._composer.close()
        super().closeEvent(event)


def _overview_guided_qss(palette, opacity):
    """Tick boxes for the guided export's lists, in the theme's colours."""
    views = ", ".join(f"QDialog#OverviewGuidedExport {kind}::indicator" for kind in ("QListView", "QTreeView"))
    checked = ", ".join(f"QDialog#OverviewGuidedExport {kind}::indicator:checked" for kind in ("QListView", "QTreeView"))
    return (f"{views} {{ width: 14px; height: 14px; border: 1px solid {palette['border']}; border-radius: 3px; "
            f"background: {palette['surface_alt']}; }} "
            f"{checked} {{ background: {palette['accent']}; border-color: {palette['accent']}; }}")


def _overview_export_qss(palette, opacity):
    """The docked export section's surface, the canvas drag handle and ticks on picked rows.

    :param palette: the active palette with rendered surface roles.
    :param opacity: the page-opacity preference, passed through unchanged.
    """
    from ..theme import RADIUS, block_surface

    surface = block_surface("surface_alt", palette["theme"], opacity)
    return (f"QFrame#OverviewExportDock {{ background: {surface}; border: 1px solid {palette['border']}; "
            f"border-radius: {RADIUS['md']}px; }}\n"
            "QFrame#OverviewExportDock QDialog#OverviewGuidedExport, "
            "QFrame#OverviewExportDock QDialog#CanvasComposer, "
            "QFrame#OverviewExportDock QWidget#CanvasComposerSettings { background: transparent; }\n"
            f"QToolButton#OverviewCanvasHandle {{ background: {palette['button_accent']}; "
            f"color: {palette['button_accent_ink']}; border: none; border-radius: {RADIUS['sm']}px; "
            "padding: 2px 8px; }\n"
            f"QToolButton#OverviewCanvasHandle:hover {{ background: {palette['button_accent_hi']}; }}\n"
            "QDialog#OverviewGuidedExport QTreeView::indicator:checked:selected "
            f"{{ background: {palette['button_accent_ink']}; border-color: {palette['button_accent_ink']}; }}")


def _overview_guided_frame(source):
    """The source's image-level measurements, or ``None`` before they were exported.

    The measurement database is looked for first, so a source never measured
    does not import the measurement module at all.
    """
    if not (Path(source) / "measurements" / "measurements.db").is_file():
        return None
    try:
        return _overview_comparison_frame(source)
    except (_OverviewValidationError, OSError, ValueError):
        return None


_OVERVIEW_UI_SOURCES = frozenset({
    *(text for caption, _key, tooltip in _OVERVIEW_SETTINGS_ROWS for text in (caption, tooltip)),
    *(text for _title, _key, caption, tooltip in _OVERVIEW_CHANNEL_TARGET_ROWS for text in (caption, tooltip)),
    *_OVERVIEW_NORMALIZATION_SCOPE_ROW,
    *_OVERVIEW_CUSTOM_COLORS_ROW,
    *(caption for _mode, caption in _OVERVIEW_COLOR_MODE_CAPTIONS),
    *(text for _name, caption, tooltip in _OVERVIEW_GLOBAL_LAYERS for text in (caption, tooltip)),
    *(caption for _value, caption in _OVERVIEW_SCOPE_CHOICES),
    *(text for row in _OVERVIEW_GUIDED_WELL_MODES + _OVERVIEW_GUIDED_FIELD_MODES for text in row),
    *_OVERVIEW_GUIDED_CAPTIONS.values(),
    *_OVERVIEW_GUIDED_SIDE_CAPTIONS.values(),
    "Guided export…",
    "Choose wells and fields, place each well on the page and choose what is printed around it, "
    "then preview and write a PDF.",
    "Custom export…",
    "Draw text boxes and drop images on a page grid, then write a PDF.",
    "Contact sheet PDFs",
    "Twelve fields per page from the export scope, with the display settings printed above them.",
    "Add selected",
    "Place the selected fields, or the fields of the selected wells, on this page.",
    "Add group",
    "Add a group with the values chosen above; columns left at Any are pooled.",
    "Remove group",
    "Remove the selected group.",
    "Move up",
    "Move the selected group earlier.",
    "Move down",
    "Move the selected group later.",
    "Name group",
    "Give the selected group a name; its images and representative are shown under that name.",
    "Clear groups",
    "Remove every group; comparisons then use the ticked values.",
    "Plate",
    "Plate shown in the plate view.",
    "Row",
    "Restrict the plate view to one row, or show every row.",
    "Column",
    "Restrict the plate view to one column, or show every column.",
    "Field",
    "Field of view shown for each well.",
    "Time",
    "Time point shown for time-lapse acquisitions.",
    "Z plane shown when the Z mode is a single plane.",
    "Reference group",
    "Group whose images define the advanced normalization bounds.",
    "Channel to edit",
    "Channel the per-channel settings are edited for: the advanced normalization, the percentile stretch, "
    "every enhancement step and the CMY/CMYK ink intensity. Pick a channel, change its values and press "
    "Apply settings to channel; channels marked (own settings) have their own values, the others follow "
    "All channels.",
    "Annotation dimensions",
    "Tick annotation columns and the values to compare; every combination of the ticked values is one group.",
    "Group values",
    "Value, or Any, of each annotation column in the next group; Any pools that column.",
    "Groups",
    "Groups compared, in order, with their labels and image counts; add, remove and reorder them here.",
    "Group presets",
    "Fill the groups with every combination of the ticked values, or vary one column and hold the others.",
    "Compare within",
    "Compare groups separately within each value of this annotation column.",
    "Measurement",
    "Image-level measurement to compare; leave empty to use the first measured column.",
    "Comparison actions",
    "Draw the group comparison figure and tables, or show one representative image per group.",
    "Selection",
    "Clear every selected field and well.",
    "PSF deconvolution",
    "Make Masks' calibrated or measured point spread function, applied to Original images after "
    "background subtraction.",
    "Cellpose 3 restoration",
    "Make Masks' Cellpose 3 denoising, deblurring or one-click restoration, applied to Original images; "
    "the model is installed and loaded through the Model Zoo.",
    "Letter",
    "Rows of the page grid.",
    "Columns of the page grid.",
    *(caption for _value, caption in _OVERVIEW_PLATES_VIEWS),
    _OVERVIEW_PLATES_VIEW_TOOLTIP,
    "Select",
    "Deselect",
    "Select all fields",
    "Deselect all fields",
})
"""Captions this module passes to ``tr`` through tuples, dicts and loop variables.

The runtime catalog generator cannot see them at a literal call site and
imports this set instead, as it does ``_DOSE_RESPONSE_UI_SOURCES``.
"""
