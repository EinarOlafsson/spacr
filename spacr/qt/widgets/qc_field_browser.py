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
from dataclasses import dataclass
from dataclasses import field as _dc_field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QStyledItemDelegate,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
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
        self._previous.setToolTip(tr("Previous flagged field (Left arrow)"))
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
        self._next.setToolTip(tr("Next flagged field (Right arrow)"))
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

    @property
    def identity(self):
        """Keep the original paths in the identity, including unresolved fields."""
        return (self.plate, self.well, self.field_id or self.field,
                self.files, self.time_id, self.z_id)

    @property
    def caption(self):
        """Trace a view back to its plate, well, field and source files."""
        return " / ".join((self.plate, self.well, self.field))


def _overview_natural_key(value):
    """Order physical field and axis identifiers by their numeric components."""
    import re
    return tuple((1, int(part)) if part.isdigit() else (0, part.casefold())
                 for part in re.split(r"(\d+)", str(value)))


def _overview_index(source, pattern="", convention="auto", cancelled=None, source_scope="recursive"):
    """Index paths without decoding pixels; never invent an unresolved well."""
    import re
    from ...schema import parse_well, well_id
    from ...regex_infer import _metadata_pattern
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
    groups = {}
    for directory, folders, names in os.walk(root):
        folders[:] = (sorted(p for p in folders if not p.startswith(".")
                            and p.lower() not in ("results", "measurements"))
                      if source_scope == "recursive" else [])
        for name in sorted(names):
            if cancelled is not None and cancelled.is_set():
                return ()
            path = Path(directory) / name
            if path.suffix.lower() not in (".tif", ".tiff", ".npy", ".png", ".jpg", ".jpeg"):
                continue
            values = {}
            if custom is not None:
                match = custom.match(name)
            else:
                match = re.match(_metadata_pattern(convention, path.suffix[1:]), name, re.I)
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
                plate = str(Path(*scope))
            well = str(values.get("wellID") or "").upper()
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
                                     field, str(time), str(plane), axes))
    return tuple(result)


def _overview_channel_info(field):
    """Inspect declared image axes and channel IDs without decoding pixels."""
    from PIL import Image

    records = []
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
            with tifffile.TiffFile(path) as image:
                series = image.series[0]
                shape, dtype, axes = series.shape, series.dtype, series.axes
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
                            z_count=shape[axes.index("Z")] if "Z" in axes else 1))
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
            page_key = record["path"], page.index
            if page_key not in page_cache:
                page_cache.clear()
                if page.is_memmappable:
                    plane = tifffile.memmap(record["path"], page=page.index, mode="r")
                else:
                    if math.prod(page.shape) * np.dtype(page.dtype).itemsize > 128 * 1024 ** 2:
                        raise _OverviewValidationError(tr("This image exceeds the 128 MiB preview decode limit."))
                    plane = page.asarray()
                page_cache[page_key] = plane
            plane = page_cache[page_key]
            selector = tuple(positions[axis] if axis in "TCZ" else slice(None)
                             for axis in page_axes)
            plane = _overview_plane_region(plane[selector], region)
            if edge is not None:
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


def _overview_pixels(field, channels=(0,), percentiles=(1., 99.), edge=1200, *,
                     normalization=None, time_index=None, z_mode="plane", z_index=None,
                     channel_views=False, normalization_context=None, cancelled=None,
                     advanced_normalization=None, region=None):
    """Decode canonical display pixels, optionally returning individual channel views."""
    from PIL import Image, ImageDraw

    planes = []
    columns = math.ceil(math.sqrt(len(channels))) if len(channels) > 3 else 1
    source_planes = []
    total_bytes = 0
    axes = dict(time_index=time_index, z_mode=z_mode, z_index=z_index)
    read_edge = None if region is not None else edge
    def read_selected(target, selected, edge=read_edge):
        """Read matching physical planes, optionally cropped in original pixels."""
        for channel, path, values in _overview_raw_planes(target, selected, edge=edge,
                region=region if edge is None else None, **axes):
            yield channel, path, values
    if region is not None and normalization is None and advanced_normalization is None:
        bounds = {}
        for channel, path, raw in read_selected(field, channels, edge=1024):
            finite = np.asarray(raw)[np.isfinite(raw)]
            if finite.size:
                bounds[channel] = (float(finite.min()), float(finite.max())) if percentiles is None else tuple(np.percentile(finite, percentiles))
        normalization = dict(mode="shared", bounds=bounds, gains={}, percentiles=percentiles)
    for item in read_selected(field, channels):
        total_bytes += item[2].nbytes
        if total_bytes > 128 * 1024 ** 2:
            raise _OverviewValidationError(tr("This image exceeds the 128 MiB preview decode limit."))
        source_planes.append(item)
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
                + (16 if len(channels) > 3 else 0)) * columns
    tile = max(1, edge // columns)
    header = min(16, tile // 4) if len(channels) > 3 else 0
    plane_edge = max(1, tile - header)
    for channel, path, array in source_planes:
        stride = max(1, math.ceil(max(array.shape) / plane_edge))
        plane = np.array(array[::stride, ::stride], dtype=np.float32, copy=True)
        finite = np.isfinite(plane)
        if not finite.any():
            raise _OverviewValidationError(tr("This channel has no finite pixels."))
        if normalization is not None or advanced_normalization is not None:
            low, high = 0., 1.
        elif percentiles is None:
            low, high = float(plane[finite].min()), float(plane[finite].max())
        else:
            low, high = np.percentile(plane[finite], percentiles)
        plane = np.nan_to_num((plane - low) / max(float(high - low), np.finfo(np.float32).eps),
                              nan=0., posinf=1., neginf=0.)
        planes.append(np.uint8(np.clip(plane, 0, 1) * 255))
    if not planes or any(plane.shape != planes[0].shape for plane in planes):
        raise _OverviewValidationError(tr("Selected channels must have matching image sizes."))
    if channel_views:
        return tuple((channel, np.repeat(plane[..., None], 3, axis=2))
                     for channel, plane in zip(channels, planes))
    if len(planes) == 1:
        return np.repeat(planes[0][..., None], 3, axis=2)
    if len(planes) > 3:
        rows = math.ceil(len(planes) / columns)
        mosaic = Image.new("RGB", (columns * tile, rows * tile))
        draw = ImageDraw.Draw(mosaic)
        for index, (channel, plane) in enumerate(zip(channels, planes)):
            x, y = (index % columns) * tile, (index // columns) * tile
            mosaic.paste(Image.fromarray(plane).convert("RGB"), (x, y + header))
            if header:
                draw.text((x, y), str(channel), fill="white")
        return np.asarray(mosaic).copy()
    rgb = np.zeros(planes[0].shape + (3,), dtype=np.uint8)
    for index, plane in enumerate(planes):
        rgb[..., index] = plane
    return rgb


def _overview_measurement(field, channels, cancelled=None, *, time_index=None,
                          z_mode="plane", z_index=None, conditions=None):
    """Measure one whole-image ROI; intensity statistics exclude nonfinite values."""
    import json
    from concurrent.futures import CancelledError
    from skimage.measure import regionprops_table

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
            values = np.asarray(plane[finite], dtype=np.float64)
            percentiles = np.percentile(values, (1, 5, 25, 50, 75, 95, 99))
            mean = float(np.mean(values))
            std = float(np.std(values))
            median = float(percentiles[3])
            stats = dict(intensity_min=float(values.min()),
                         intensity_max=float(values.max()), intensity_mean=mean,
                         intensity_sum=float(values.sum()), intensity_std=std,
                         intensity_variance=float(np.var(values)),
                         intensity_median=median,
                         intensity_mad=float(np.median(np.abs(values - median))),
                         intensity_iqr=float(percentiles[4] - percentiles[2]),
                         intensity_zero_fraction=float(np.count_nonzero(values == 0) / count))
            if std:
                centred = (values - mean) / std
                stats.update(intensity_skewness=float(np.mean(centred ** 3)),
                             intensity_kurtosis=float(np.mean(centred ** 4) - 3))
            else:
                stats.update(intensity_skewness=0., intensity_kurtosis=0.)
            stats.update({"intensity_p" + str(p): float(v)
                          for p, v in zip((1, 5, 25, 50, 75, 95, 99), percentiles)})
            record.update({prefix + key: value for key, value in stats.items()})
        if shape is None:
            raise ValueError(tr("Choose one or more nonnegative channel indices."))
    except CancelledError:
        raise
    except Exception as error:
        record.update(status="error", error=str(error))
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


def _overview_measurements_export(fields, source, channels, workers=1,
                                  cancelled=None, annotations=None, progress=None, *,
                                  time_index=None, z_mode="plane", z_index=None, field_options=None):
    """Publish an atomic raw-pixel FOV run, with one bounded image per worker.

    ``progress(completed, total)`` runs in the caller thread. Failed or cancelled
    runs leave existing measurements intact and return per-field diagnostics.
    Selected channels replace the previous row for the same source field.
    """
    import hashlib
    import json
    import multiprocessing
    import sqlite3
    import tempfile
    from concurrent.futures import CancelledError, ProcessPoolExecutor, wait, FIRST_COMPLETED
    from datetime import datetime, timezone
    from ...database_concurrency import connect, transaction
    from ...schema import row_id, column_id, field_id, time_id, compose_prc, compose_prcf

    source = Path(source).expanduser().resolve()
    destination = source / "measurements" / "measurements.db"
    channels = tuple(dict.fromkeys(int(value) for value in channels))
    if not channels or any(value < 0 for value in channels):
        raise ValueError(tr("Choose one or more nonnegative channel indices."))
    workers = max(1, min(int(workers), os.cpu_count() or 1))
    fields = tuple(fields)
    report = dict(path=str(destination), count=0, errors=[], cancelled=False,
                  attempted=0, diagnostics=[], workers=workers)
    if not fields:
        return report
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

    executor = None
    pending = {}
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
                            **axes_for(field)), stage)
                else:
                    executor = ProcessPoolExecutor(max_workers=workers,
                        mp_context=multiprocessing.get_context("spawn"))
                    remaining = iter(fields)
                    for field in remaining:
                        check_cancelled()
                        pending[executor.submit(_overview_measurement, field, channels, **axes_for(field))] = field
                        if len(pending) >= workers:
                            break
                    while pending:
                        check_cancelled()
                        done, _ = wait(pending, timeout=.1, return_when=FIRST_COMPLETED)
                        for future in done:
                            field = pending.pop(future)
                            try:
                                row = future.result()
                            except Exception as error:
                                row = dict(status="error", error=str(error))
                            record_result(field, row, stage)
                            next_field = next(remaining, None)
                            if next_field is not None:
                                check_cancelled()
                                pending[executor.submit(_overview_measurement,
                                    next_field, channels, **axes_for(next_field))] = next_field
                check_cancelled()
                if report["errors"]:
                    return report
                destination.parent.mkdir(parents=True, exist_ok=True)
                connection = connect(destination, timeout=30)
                try:
                    with transaction(connection, attempts=8, busy_timeout=30):
                        check_cancelled()
                        connection.execute('CREATE TABLE IF NOT EXISTS fov (field_uid TEXT PRIMARY KEY)')
                        existing = {row[1] for row in connection.execute('PRAGMA table_info(fov)')}
                        if "field_uid" not in existing:
                            connection.execute('ALTER TABLE fov ADD COLUMN field_uid TEXT')
                            existing.add("field_uid")
                        connection.execute('CREATE UNIQUE INDEX IF NOT EXISTS fov_field_uid ON fov(field_uid)')
                        for (serialized,) in stage.execute("SELECT record FROM records"):
                            check_cancelled()
                            row = json.loads(serialized)
                            for key, value in row.items():
                                if key not in existing:
                                    kind = ("INTEGER" if isinstance(value, int) else
                                            "REAL" if isinstance(value, float) else "TEXT")
                                    connection.execute(f'ALTER TABLE fov ADD COLUMN "{key}" {kind}')
                                    existing.add(key)
                            connection.execute('DELETE FROM fov WHERE field_uid = ?', (row["field_uid"],))
                            names = ", ".join('"' + key + '"' for key in row)
                            placeholders = ", ".join("?" for _ in row)
                            connection.execute(f'INSERT INTO fov ({names}) VALUES ({placeholders})',
                                               tuple(row.values()))
                        check_cancelled()
                    report["count"] = report["attempted"]
                finally:
                    connection.close()
            finally:
                stage.close()
    except CancelledError:
        report["cancelled"] = True
    finally:
        if executor is not None:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
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


def _overview_pdf_header(channels, percentiles, seed, normalization):
    """Describe the selected display transform and its actual sampled reference."""
    import textwrap

    lines = [tr("Channels: {channels}; percentiles: {percentiles}; seed: {seed}",
                channels=str(channels), percentiles=str(percentiles), seed=seed)]
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


def _overview_export(fields, folder, channels, percentiles, seed=0, cancelled=None,
                     measurements=False, *, quality_edge=1000, dpi=None,
                     normalization=None, time_index=None, z_mode="plane", z_index=None,
                     conditions=None, normalization_context=None, annotations=None,
                     advanced_normalization=None, field_options=None):
    """Write twelve bounded images per styled PDF with reproducible provenance."""
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
            None if advanced_normalization is not None else page_normalization)
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
                        cancelled=cancelled,
                        channel_views=True, advanced_normalization=advanced_normalization, **selected_axes)
                    channel_columns = math.ceil(math.sqrt(len(views)))
                    channel_rows = math.ceil(len(views) / channel_columns)
                    inner = panel.subgridspec(channel_rows, channel_columns, hspace=.18, wspace=.08)
                    for position, (channel, pixels) in enumerate(views):
                        axis = figure.add_subplot(inner[position // channel_columns, position % channel_columns])
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
                                              cancelled=cancelled, advanced_normalization=advanced_normalization, **selected_axes)
                    axis = figure.add_subplot(panel)
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
                "field_options": field_options,
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
                      normalization_context=None, advanced_normalization=None, preview_edge=240):
    """Render one bounded page of field thumbnails, using the shared pixel path."""
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
        draw.text((x, y), field.well + " / " + field.field, fill="white")
        try:
            pixels = _overview_pixels(field, channels, percentiles, edge=preview_edge,
                                      normalization=normalization,
                                      normalization_context=normalization_context,
                                      cancelled=cancelled, time_index=time_index,
                                      z_mode=z_mode, z_index=z_index,
                                      advanced_normalization=advanced_normalization)
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
        from PySide6.QtCore import QRect
        from PySide6.QtGui import QColor, QPalette
        from PySide6.QtWidgets import QStyle, QStyleOptionViewItem

        opts = QStyleOptionViewItem(option)
        self.initStyleOption(opts, index)
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


class _FieldOverview(QWidget):
    """Shared, lazily loaded plate and field overview for both mask modules."""

    def __init__(self, parent=None, settings=None):
        """Create controls without opening a directory or decoding an image.

        :param parent: the module that owns this retained browser.
        :param settings: accepted metadata, geometry and display configuration.
        """
        from collections import OrderedDict
        from PySide6.QtWidgets import (QLineEdit, QSpinBox, QDoubleSpinBox,
                                       QListWidget, QAbstractItemView, QTableWidget)
        import threading

        super().__init__(parent)
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
        self._annotations = {}
        self._metadata_revision = 0
        self._annotation_write_lock = threading.Lock()
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
        self._geometry_valid = True
        self._geometry_error = None
        self._plate_cell_extent = None
        self._plate_fit = True
        self._displayed_field = None
        self.setWindowTitle(tr("Field-of-view overview"))
        self.resize(1000, 760)
        self._fields, self._visible = (), ()
        self._selected_fields = {}
        self._cache = OrderedDict()
        self._thumbnail_cache = OrderedDict()
        self._well_representatives = {}
        self._well_positions = {}
        self._installed_thumbnails = set()
        self._individual = False
        self._visual_signature = None
        self._displayed_montage = None
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
        self._thumbnail_jobs.job_finished.connect(self._request_visuals)
        self._montage_jobs.job_finished.connect(self._request_visuals)
        self._scope = (None, None)
        self._image_jobs.job_finished.connect(self._refresh_image_if_pending)
        self._image_pending = False
        self._page = 0
        self._comparison_signature = None
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
        self.normalise = QCheckBox(tr("Percentile normalization"))
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
        self._navigation_layout = navigation
        column.addLayout(navigation)
        from PySide6.QtCore import QSize
        self.wells, self.fields = QTableWidget(), QListWidget()
        self.wells.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.wells.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.wells.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.wells.horizontalHeader().setMinimumSectionSize(1)
        self.wells.verticalHeader().setMinimumSectionSize(1)
        self.wells.setItemDelegate(_OverviewPlateDelegate(self.wells))
        self.wells.setMouseTracking(True)
        self.wells.viewport().setMouseTracking(True)
        self.fields.setSelectionMode(QAbstractItemView.ExtendedSelection)
        from PySide6.QtWidgets import QListView
        self.fields.setViewMode(QListView.IconMode)
        self.fields.setResizeMode(QListView.Adjust)
        self.fields.setMovement(QListView.Static)
        self.fields.setSelectionRectVisible(True)
        self.fields.setWrapping(True)
        self.fields.setWordWrap(True)
        self.fields.setIconSize(QSize(128, 96))
        self.fields.setGridSize(QSize(154, 140))
        self.fields.viewport().installEventFilter(self)
        self.fields.verticalScrollBar().valueChanged.connect(self._request_visuals)
        self._well_viewport = self.wells.viewport()
        self._well_viewport.installEventFilter(self)
        self.wells.installEventFilter(self)
        self.wells.horizontalScrollBar().valueChanged.connect(self._request_visuals)
        self.wells.verticalScrollBar().valueChanged.connect(self._request_visuals)
        from PySide6.QtCore import QSize
        self.wells.setIconSize(QSize(160, 120))
        follow_device_ratio(self.wells.viewport(), self._request_visuals)
        self.wells.itemDoubleClicked.connect(self._well)
        self.wells.itemClicked.connect(self._clicked_well)
        self.wells.itemSelectionChanged.connect(self._remember_wells)
        self.wells.setToolTip(tr("Select wells for export; double-click a well to explore its fields."))
        self.fields.itemSelectionChanged.connect(self._remember_selection)
        self.fields.itemClicked.connect(self._field)
        self.fields.currentItemChanged.connect(self._current_field_changed)
        from PySide6.QtWidgets import QSplitter
        splitter = QSplitter()
        self._splitter = splitter
        splitter.addWidget(self.wells)
        splitter.addWidget(self.fields)
        self.view = _OverviewImageView()
        self.view.axis_step.connect(self._step_axis)
        self.view.detail_requested.connect(self._request_detail)
        self.view.horizontalScrollBar().valueChanged.connect(self._request_native_detail)
        self.view.verticalScrollBar().valueChanged.connect(self._request_native_detail)
        splitter.addWidget(self.view)
        splitter.setStretchFactor(2, 3)
        column.addWidget(splitter, 1)
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
        for value in (128, 256, 512, 1024, 2048, 4096):
            self.preview_resolution.addItem(str(value), value)
        self.preview_resolution.setCurrentIndex(3)
        self.preview_resolution.setToolTip(tr(
            "Maximum preview edge in pixels; lower values load faster. Zoom still loads original pixels."))
        self.preview_resolution.currentIndexChanged.connect(self._resolution_changed)
        axes.addWidget(QLabel(tr("Preview resolution")))
        axes.addWidget(self.preview_resolution)
        compare = QPushButton(tr("Compare selected fields"))
        compare.clicked.connect(self._compare)
        axes.addWidget(compare)
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
        self.export_scope = QComboBox()
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
        self.random = QCheckBox(tr("Random sample"))
        self.use_selection = QCheckBox(tr("Export selected fields or wells"))
        self.sample_selected = QCheckBox(tr("Sample within selected wells"))
        self.measurements = QCheckBox(tr("Measurements CSV (raw image pixels)"))
        self.measurements.setToolTip(tr(
            "Measure original-resolution selected channels as one whole-image ROI, not segmented objects. "
            "Geometry is in pixels; intensity statistics use finite stored pixel values without display "
            "normalization. Nonfinite pixel counts and read errors are recorded."))
        export, cancel = QPushButton(tr("Export PDFs")), QPushButton(tr("Cancel loading or export"))
        clear_selection = QPushButton(tr("Clear selections"))
        clear_selection.clicked.connect(self._clear_selection)
        export.clicked.connect(self._export)
        cancel.clicked.connect(self._cancel)
        for widget in (previous, following):
            paging.addWidget(widget)
        self._plate_scale = 1.
        for text, step in ((tr("Fit plate"), 0), (tr("Larger wells"), 1), (tr("Smaller wells"), -1)):
            button = QPushButton(text)
            button.clicked.connect(lambda _checked=False, value=step: self._zoom_plate(value))
            paging.addWidget(button)
        column.addLayout(paging)
        sample_row = QHBoxLayout()
        for widget in (self.grouping, QLabel(tr("Samples per group")), self.count,
                       QLabel(tr("Seed")), self.seed, self.random):
            sample_row.addWidget(widget)
        column.addLayout(sample_row)
        export_row = QHBoxLayout()
        for widget in (self.use_selection, self.sample_selected, clear_selection, export, cancel):
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
        self._module_actions = QWidget(self)
        action_row = QHBoxLayout(self._module_actions)
        for caption, callback in ((tr("Export PDFs"), self._export),
                (tr("Export image-level measurements"), self._measure),
                (tr("Cancel loading or export"), self._cancel)):
            button = QPushButton(caption)
            button.clicked.connect(callback)
            action_row.addWidget(button)
        self._module_actions.hide()

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
        options.update(scope=self.normalization_scope.currentData(),
                       per_channel={str(key): dict(value) for key, value in self._channel_overrides.items()})
        return options

    def _normalization_target_changed(self, *_):
        """Load the effective settings for one channel without changing others."""
        if self._editing_channel is None:
            self._global_normalization_options = self._editor_options()
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
        """Commit editor settings to exactly the selected channel."""
        channel = self.norm_channel.currentData()
        if channel is None:
            self._global_normalization_options = self._editor_options()
        else:
            self._channel_overrides[int(channel)] = self._editor_options()
        self._advanced_invalidate()

    def _clear_channel_normalization(self):
        """Reset channel overrides and reload the retained global transform."""
        self._channel_overrides.clear()
        self._normalization_target_changed()
        self._advanced_invalidate()

    def _build_advanced_controls(self, column):
        """Keep independent channel editing distinct from the existing reference modes."""
        from PySide6.QtWidgets import QSpinBox, QDoubleSpinBox, QGridLayout
        from .setup_dialog import _SetupComboBox
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        self.advanced = QCheckBox(tr("Advanced channel normalization"))
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
        self.norm_channel = QComboBox()
        self.norm_channel.addItem(tr("All channels"), None)
        self._editing_channel = None
        self._global_normalization_options = self._editor_options()
        apply = QPushButton(tr("Apply settings to channel"))
        apply.clicked.connect(self._apply_channel_normalization)
        reset = QPushButton(tr("Reset channel overrides"))
        reset.clicked.connect(self._clear_channel_normalization)
        override_host = FlowHost(widget)
        row = FlowLayout(override_host)
        for control in (self.norm_channel, apply, reset):
            row.addWidget(control)
        layout.addWidget(override_host)
        column.addWidget(widget)
        channel_host = FlowHost(self)
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
        payload = dict(wells=[dict(plate=plate, well=well, metadata=metadata)
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
        def read():
            """Validate source-relative metadata without GUI-thread file access."""
            import json
            path = Path(source) / "results" / "overview_annotations.json"
            if not path.is_file():
                return {}
            records = json.loads(path.read_text(encoding="utf-8")).get("wells", ())
            values = {}
            for record in records:
                address = str(record["plate"]), str(record["well"])
                if address not in known or address in values or not isinstance(record["metadata"], dict):
                    raise _OverviewValidationError(tr("Annotation table identities are missing, duplicated or outside this source."))
                values[address] = {str(key): str(value) for key, value in record["metadata"].items()}
            return values
        def loaded(values):
            """Install only metadata belonging to the unchanged indexed generation."""
            if cancelled.is_set() or fields is not self._fields or revision != self._metadata_revision:
                return
            for address, metadata in values.items():
                self._annotations.setdefault(address, {}).update(metadata)
                if metadata.get("condition") and address not in self._conditions:
                    self._conditions[address] = metadata["condition"]
            self._fill_group_choices(self.grouping)
            self._fill_group_choices(self.normalization_scope, image=True)
            self._filter()
            if self.advanced.isChecked():
                self._prepare_display()
        self._metadata_jobs.submit(read, loaded)

    def _remember_wells(self):
        """Retain plate-qualified selections and each plate's acquisition context."""
        plate = self.plates.currentData()
        for well in self._well_positions:
            self._selected_wells.discard((plate, well))
            self._selected_well_axes.pop((plate, well), None)
            self._selected_well_options.pop((plate, well), None)
        for item in self.wells.selectedItems():
            address = plate, item.data(Qt.UserRole)
            self._selected_wells.add(address)
            self._selected_well_axes[address] = (self.time_axis.currentData(), self.z_axis.currentData())
            self._selected_well_options[address] = self._axis_options()

    def _clicked_well(self, item):
        """Open ordinary clicks while retaining drag and modifier selections."""
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
        for channel in available:
            control = QCheckBox(str(channel))
            control.setChecked(channel in selected)
            control.setToolTip(tr("Show channel {channel}", channel=channel))
            control.toggled.connect(self._channel_toggled)
            self._channel_bar.addWidget(control)
            self._channel_buttons[channel] = control
            self.norm_channel.addItem(str(channel), channel)
        self.norm_channel.blockSignals(False)
        self._normalization_target_changed()

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
        source, workers = self._metadata_source, self.workers.value()
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
                self.status.setText(tr("Saved {count} image measurements to {path} (table fov).",
                    count=report["count"], path=report["path"]))
        self.status.setText(tr("Measuring {count} image planes with {workers} parallel workers…", count=len(fields), workers=workers))
        self._measurement_jobs.submit(lambda: _overview_measurements_export(fields, source, channels,
            workers, cancelled, annotations, **axes), done)

    def _save_annotations(self):
        """Write conditions only to the user's explicitly chosen output table."""
        from PySide6.QtWidgets import QFileDialog
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
                self._filter()
                if self._overview_options.get("normalization") or self.advanced.isChecked():
                    self._prepare_display()
        self._export_jobs.submit(read, loaded)

    def _update_field_thumbnails(self, signature, channels, limits, options):
        """Decode one visible field tile at a time, with at most 24 native icons.

        :param signature: the current source, axes and display generation.
        :param channels: exact requested source channel identities.
        :param limits: optional percentile limits.
        :param options: the immutable normalization and acquisition selection.
        """
        if self._field_thumbnail_jobs.is_busy():
            return
        viewport, cancelled = self.fields.viewport(), self._cancelled
        ratio = device_ratio(viewport)
        edge = min(math.ceil(128 * ratio), self.preview_resolution.currentData())
        for index in range(self.fields.count()):
            item = self.fields.item(index)
            if not viewport.rect().intersects(self.fields.visualItemRect(item)):
                continue
            field = item.data(Qt.UserRole)
            key = field.identity, channels, limits, self._display_options_key(options), ratio, edge
            if item.data(Qt.UserRole + 1) == key:
                continue
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
                if cancelled.is_set() or self._visual_context() != signature:
                    return
                pixels, error = result
                if error:
                    self.status.setText(tr("Could not read {field}: {error}", field=field.caption, error=error))
                for current_index in range(self.fields.count()):
                    current = self.fields.item(current_index)
                    if current.data(Qt.UserRole).identity != field.identity:
                        continue
                    current.setData(Qt.UserRole + 1, key)
                    if pixels is None:
                        current.setIcon(QIcon())
                    else:
                        image = QImage(pixels.data, pixels.shape[1], pixels.shape[0],
                                       pixels.strides[0], QImage.Format_RGB888).copy()
                        pixmap = scaled_for(QPixmap.fromImage(image), viewport, self.fields.iconSize())
                        current.setIcon(QIcon(pixmap))
                    break
            self._field_thumbnail_jobs.submit(read, shown)
            return

    def _zoom_plate(self, delta=0):
        """Fit the complete physical plate or explicitly magnify its cells.

        :param delta: signed magnification step, or zero to restore whole-plate fit.
        """
        self._plate_fit = not bool(delta)
        columns, rows = max(1, self.wells.columnCount()), max(1, self.wells.rowCount())
        if delta and self._plate_cell_extent is not None:
            factor = 1.25 if delta > 0 else .8
            width, height = (max(2, min(512, round(value * factor)))
                             for value in self._plate_cell_extent)
        else:
            width = max(1, self.wells.viewport().width() // columns)
            height = max(1, self.wells.viewport().height() // rows)
        self.wells.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff if self._plate_fit else Qt.ScrollBarAsNeeded)
        self.wells.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff if self._plate_fit else Qt.ScrollBarAsNeeded)
        self._plate_cell_extent = width, height
        self._apply_plate_extent()

    def _apply_plate_extent(self):
        """Restore the same native well size after source-axis or annotation changes."""
        width, height = self._plate_cell_extent
        from PySide6.QtCore import QSize
        self.wells.setIconSize(QSize(max(1, width - 2), max(1, height - 2)))
        for column in range(self.wells.columnCount()):
            self.wells.setColumnWidth(column, width)
        for row in range(self.wells.rowCount()):
            self.wells.setRowHeight(row, height)
        self._request_visuals()

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
        from .card import Card
        self._options_window = QDialog(self, Qt.Window)
        self._options_window.setWindowTitle(tr("Options"))
        self._options_window.setWindowModality(Qt.NonModal)
        self._options_window.resize(880, 680)
        options_layout = QVBoxLayout(self._options_window)
        content = QWidget(self._options_window)
        self._settings_content = content
        self._settings_host = None
        controls = QVBoxLayout(content)
        categories = {}
        for name in (tr("Source"), tr("Display"), tr("Annotations"), tr("Export")):
            card = Card(title=name, foldable=True)
            controls.addWidget(card)
            categories[name] = card.body_layout
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
                    categories[category].addWidget(host)
            else:
                categories[category].addWidget(item.widget())
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
        column.addWidget(essential)
        column.addWidget(splitter, 1)
        column.addWidget(self.status)
        self.fields.hide()
        self.view.hide()

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
        return repr(values), id(advanced) if advanced is not None else None

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

    def _display_kwargs(self):
        """Freeze selected stack axes and the immutable normalization plan."""
        time = self.time_axis.currentData()
        z = self.z_axis.currentData()
        return dict(normalization=self._normalization_plan,
                    normalization_context=self._normalization_context,
                    time_index=time if isinstance(time, int) else 0,
                    z_index=z if isinstance(z, int) else 0,
                    z_mode=self.z_mode.currentData() or "plane",
                    advanced_normalization=self._advanced_model if self.advanced.isChecked() else None)

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
        """Rebuild bounded previews without changing raw data or learned display limits."""
        self._cache.clear()
        self._thumbnail_cache.clear()
        self._displayed_montage = None
        self._displayed_field = None
        self._request_visuals()
        item = self.fields.currentItem()
        if self._individual and item is not None:
            self._field(item)

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
        if not self._individual or self.view._item is None:
            return
        width = self.view._item.pixmap().width()
        displayed = width * self.view.transform().m11()
        desired = min(4096, max(256, int(math.ceil(displayed * device_ratio(self.view.viewport())))))
        desired = min(self.preview_resolution.currentData(),
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
        """Overlay a bounded source-resolution crop in the retained preview scene."""
        if not self._individual or self._cancelled.is_set() or self.view._item is None:
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
        if len(channels) > 3:
            self.status.setText(tr("For original-resolution detail, toggle up to three channels or inspect one channel individually."))
            return
        preview = self.view._item.pixmap()
        height, width = shape
        scale_x, scale_y = preview.width() / width, preview.height() / height
        if self.view.transform().m11() * scale_x < .5:
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
        options, signature, cancelled = self._display_kwargs(), self._visual_context(), self._cancelled
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
        if not addresses or not key:
            self.status.setText(tr("Choose a row, column or well selection before annotating."))
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
        self._persist_metadata()
        self._filter()
        if self._overview_options.get("normalization") or self.advanced.isChecked():
            self._prepare_display()

    def _compare(self):
        """Render only explicitly selected fields, retaining their source identities."""
        fields = tuple(self._selected_fields.values())
        if not fields:
            self.status.setText(tr("Select fields by clicking or dragging a selection rectangle."))
            return
        if len(fields) > 24:
            self.status.setText(tr("Compare up to 24 fields at a time."))
            return
        try:
            channels, limits = self._choices()
        except (ValueError, TypeError) as exc:
            self.status.setText(str(exc))
            return
        options, cancelled = self._display_kwargs(), self._cancelled
        self._individual = False
        self.fields.show()
        self.view.show()
        self._splitter.setSizes([250, 300, 750])
        signature = tuple(field.identity for field in fields), channels, limits, self._display_options_key(options)
        self._comparison_signature = signature
        def done(result):
            """Show only a comparison that still owns the current navigation.

            :param result: detached montage pixels and per-field errors.
            """
            if (not cancelled.is_set() and not self._individual
                    and self._comparison_signature == signature):
                pixels, errors = result
                if pixels is not None:
                    self._show_pixels(pixels)
                    self.status.setText(tr("{count} fields; {errors} unreadable previews.",
                                           count=len(fields), errors=len(errors)))
        self._montage_jobs.submit(lambda: _overview_montage(
            fields, channels, limits, cancelled, **options), done)

    def _choices(self):
        """Validate one display/export configuration on the GUI thread."""
        try:
            channels = tuple(int(token.strip()) for token in self.channels.text().split(","))
        except (ValueError, TypeError):
            raise _OverviewValidationError(tr("Choose one or more nonnegative channel indices.")) from None
        if not channels or min(channels) < 0:
            raise _OverviewValidationError(tr("Choose one or more nonnegative channel indices."))
        limits = (self.low.value(), self.high.value()) if self.normalise.isChecked() else None
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
        self._individual = False
        self._selected_fields.clear()
        self._selected_wells.clear()
        self._selected_well_axes.clear()
        self._selected_well_options.clear()
        self._selected_field_options.clear()
        self._fields, self._visible = (), ()
        self.wells.clear()
        self.fields.clear()
        self.view.clear_image()
        source, regex, convention, cancelled = self.source.text(), self.regex.text(), self._convention, self._cancelled
        self.status.setText(tr("Indexing image metadata…"))
        source_scope = self._overview_options.get("source_scope", "recursive")
        def read():
            """Reject unresolved wells on every browser load, retaining honest previews."""
            fields = _overview_matched_fields(_overview_index(
                source, regex, convention, cancelled, source_scope=source_scope), regex)
            if any(not field.well or field.unresolved for field in fields):
                raise _OverviewValidationError(tr("The filename regex must include a valid wellID for every image."))
            return fields
        self._index_jobs.submit(read, self._indexed)

    def _indexed(self, fields):
        """Show actual plate identities and explicitly retained unresolved fields.

        :param fields: immutable source records returned by the metadata indexer.
        """
        self._fields = fields
        self._metadata_source = self.source.text().strip()
        self._restore_metadata()
        self._populate_axes(fields)
        channel_ids = sorted({channel for field in fields for channel, _ in field.files if channel},
                             key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))
        if channel_ids and self.channels.text() == "0":
            self.channels.setText(",".join(channel_ids[:3]))
        self.plates.blockSignals(True)
        self.plates.clear()
        for plate in sorted({field.plate for field in fields}):
            self.plates.addItem(plate, plate)
        options = self._overview_options.get("normalization") or {}
        reference = options.get("reference")
        if (options.get("mode") == "reference_average"
                and isinstance(reference, (tuple, list)) and len(reference) == 2):
            index = self.plates.findData(reference[0])
            if index >= 0:
                self.plates.setCurrentIndex(index)
        self.plates.blockSignals(False)
        self._back()
        if not self._geometry_valid:
            return
        self._prepare_display()
        self.status.setText(tr("{count} fields; {unresolved} with unresolved metadata.",
                               count=len(fields), unresolved=sum(f.unresolved for f in fields)))

    def _back(self):
        """Return to the current plate, retaining its identity and filters."""
        self._scope = (self.plates.currentData(), None)
        self._individual = False
        self._comparison_signature = None
        self.fields.hide()
        self.view.hide()
        self.wells.show()
        self._plate_fit = True
        self._plate_cell_extent = None
        self._splitter.setSizes([1000, 0, 0])
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

        :param _: unused plate, row or column selection arguments.
        """
        current = self.fields.currentItem()
        current_identity = current.data(Qt.UserRole).identity if current is not None else None
        plate = self.plates.currentData()
        self._comparison_signature = None
        if self._scope[0] != plate:
            self._scope = (plate, None)
        available = tuple(f for f in self._fields if f.plate == plate)
        try:
            geometry = self._checked_plate_geometry(available, self._overview_options.get("geometry"))
        except _OverviewValidationError as error:
            self._geometry_error = str(error)
            self._geometry_valid = False
            self._normalization_ready = False
            self._visible = ()
            self.wells.clear()
            self.wells.setRowCount(0)
            self.wells.setColumnCount(0)
            self.fields.clear()
            self.view.clear_image()
            self.status.setText(str(error))
            return
        self._geometry_valid = True
        self._geometry_error = None
        for combo, values, title in ((self.rows, sorted({f.row for f in available if f.row}, key=int), tr("All rows")),
                                     (self.columns, sorted({f.column for f in available if f.column}), tr("All columns"))):
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
        available = tuple(f for f in available if (row is None or f.row == row) and
                          (col is None or f.column == col))
        from collections import Counter
        from .sortable_table import table_item
        known_wells = {field.well for field in available}
        addresses = {f.well: (int(f.row), f.column) for f in available if not f.unresolved}
        rows = max((position[0] for position in addresses.values()), default=0)
        columns = max((position[1] for position in addresses.values()), default=1)
        if geometry is not None:
            rows, columns = int(geometry[0]), int(geometry[1])
        if rows > 512 or columns > 512:
            self.status.setText(tr("Plate coordinates exceed the overview grid limit."))
            return
        self._well_representatives = {}
        fov, time, z = (combo.currentData() for combo in (self.fov_axis, self.time_axis, self.z_axis))
        available = tuple(field for field in available
            if (fov is None or getattr(field, "field_id", "") == fov)
            and (not getattr(field, "time_id", "") or field.time_id == time)
            and (not getattr(field, "z_id", "") or field.z_id == z))
        counts = Counter(f.well for f in available)
        self._well_fields = {}
        for field in available:
            self._well_fields.setdefault(field.well, []).append(field)
            self._well_representatives.setdefault(field.well, field)
        self._well_positions = {}
        self._installed_thumbnails.clear()
        self.wells.blockSignals(True)
        self.wells.clear()
        coverslip = geometry == (1, 1) and not addresses and known_wells == {""}
        self.wells.setRowCount(1 if coverslip else rows + int("" in known_wells))
        self.wells.setColumnCount(columns)
        self.wells.setHorizontalHeaderLabels([str(n) for n in range(1, columns + 1)])
        from ...schema import letters_from_row_index
        self.wells.setVerticalHeaderLabels([tr("Unresolved")] if coverslip else
            [letters_from_row_index(n) for n in range(1, rows + 1)]
            + ([tr("Unresolved")] if "" in known_wells else []))
        for well in sorted(known_wells):
            count = counts[well]
            r, c = addresses.get(well, (1 if coverslip else rows + 1, 1))
            item = table_item(tr("{well}\n{count} fields", well=well or tr("Unresolved"), count=count))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            item.setData(Qt.UserRole, well)
            detail = [item.text(), self._conditions.get((plate, well), "")]
            detail.extend(str(key) + ": " + str(value)
                          for key, value in self._annotation_snapshot().get((plate, well), {}).items())
            detail.extend(field.caption for field in self._well_fields.get(well, ())
                          [:self.representatives.value()])
            if not count:
                detail.append(tr("No image at the selected acquisition axes."))
            item.setToolTip("\n".join(part for part in detail if part))
            self.wells.setItem(r - 1, c - 1, item)
            item.setSelected((plate, well) in self._selected_wells)
            self._well_positions[well] = (r - 1, c - 1)
            self.wells.setRowHeight(r - 1, 150)
        for index in range(columns):
            self.wells.setColumnWidth(index, 180)
        self.wells.blockSignals(False)
        for address in self._selected_wells:
            if address[0] == plate:
                self._selected_well_axes[address] = (time, z)
        self._visible = tuple(f for f in available if self._scope[1] is None or f.well == self._scope[1])
        self._page = min(self._page, max(0, (len(self._visible) - 1) // 24))
        self.fields.blockSignals(True)
        self.fields.clear()
        from PySide6.QtWidgets import QListWidgetItem
        from PySide6.QtCore import QItemSelectionModel
        for field in self._visible[self._page * 24:(self._page + 1) * 24]:
            item = QListWidgetItem(tr("{well} · field {field} · T {time} · Z {z}",
                well=field.well or tr("Unresolved"), field=getattr(field, "field_id", "") or field.field,
                time=getattr(field, "time_id", "") or str(self.time_axis.currentData()),
                z=getattr(field, "z_id", "") or str(self.z_axis.currentData())))
            item.setToolTip(field.caption)
            item.setData(Qt.UserRole, field)
            self.fields.addItem(item)
            item.setSelected(field.identity in self._selected_fields)
            if field.identity == current_identity:
                self.fields.setCurrentItem(item, QItemSelectionModel.NoUpdate)
        self.fields.blockSignals(False)
        if not self._visible:
            self.view.clear_image()
        if self._overview_options.get("geometry"):
            if self._plate_fit or self._plate_cell_extent is None:
                self._zoom_plate()
            else:
                self._apply_plate_extent()
        if (self._overview_options.get("normalization")
                and self._normalization_plate != plate):
            self._prepare_display()
        self._request_visuals()

    def _remember_selection(self):
        """Retain explicit field selections across pages and plate navigation."""
        for index in range(self.fields.count()):
            item = self.fields.item(index)
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
        self.fields.clearSelection()
        self.wells.clearSelection()

    def _well(self, item):
        """Zoom into one well while retaining the current plate and filters.

        :param item: the selected physical well cell.
        """
        self._scope = (self.plates.currentData(), item.data(Qt.UserRole))
        self._comparison_signature = None
        self.fields.show()
        self.view.show()
        self._splitter.setSizes([250, 200, 750])
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
        self._displayed_montage = None
        self._comparison_signature = None
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
        if not self._individual:
            self._splitter.setSizes([250, 300, 750])
        self._individual = True
        self._comparison_signature = None
        self._displayed_montage = None
        if not self._normalization_ready:
            self.view.clear_image()
            return
        options = self._display_kwargs()
        source_key = field.identity, channels, limits, self._display_options_key(options)
        if source_key != self._displayed_field:
            self._preview_edge = min(256, self.preview_resolution.currentData())
            self._displayed_field = source_key
            preserve_view = False
        edge = self._preview_edge
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
        target_edge = self.preview_resolution.currentData()
        def done(result):
            """Cache bounded pixels and accept only the exact current axis/view.

            :param result: detached pixels and the original header dimensions.
            """
            pixels, shape = result
            if cancelled.is_set():
                return
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
                if edge == min(256, target_edge) and edge < target_edge:
                    self._preview_edge = target_edge
                    self._image_pending = True
        def read():
            """Read original dimensions and a bounded preview without Qt access."""
            info = _overview_channel_info(field)
            record = info["files"][0]
            shape = tuple(record["shape"][record["axes"].index(axis)] for axis in "YX")
            return _overview_pixels(field, channels, limits, edge=edge, **options), shape
        self._image_jobs.submit(read, done)

    def eventFilter(self, watched, event):
        """Request only newly visible well images after viewport paints or resizes.

        :param watched: the actual plate or field-list viewport.
        :param event: a paint or resize event that may expose image tiles.
        :returns: the inherited event-filter result.
        """
        if watched is self._well_viewport:
            if event.type() == QEvent.MouseButtonPress:
                self._drag_start = event.position().toPoint()
                self._drag_selected = False
                self._press_modifiers = event.modifiers()
            elif event.type() == QEvent.MouseMove and self._drag_start is not None and event.buttons() & Qt.LeftButton:
                from PySide6.QtWidgets import QApplication
                self._drag_selected |= (event.position().toPoint() - self._drag_start).manhattanLength() >= QApplication.startDragDistance()
            elif event.type() == QEvent.MouseButtonRelease:
                self._drag_start = None
        if (watched is self._well_viewport and event.type() == QEvent.Wheel
                and event.angleDelta().y()):
            position = event.position().toPoint()
            old_width, old_height = self._plate_cell_extent or (1, 1)
            column = (self.wells.horizontalScrollBar().value() + position.x()) / old_width
            row = (self.wells.verticalScrollBar().value() + position.y()) / old_height
            self._zoom_plate(1 if event.angleDelta().y() > 0 else -1)
            self.wells.doItemsLayout()
            width, height = self._plate_cell_extent
            self.wells.horizontalScrollBar().setValue(round(column * width - position.x()))
            self.wells.verticalScrollBar().setValue(round(row * height - position.y()))
            event.accept()
            return True
        if watched in (self.wells, self._well_viewport) and event.type() == QEvent.KeyPress and event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._open_selected_well()
            return True
        if watched is self._well_viewport and event.type() == QEvent.Resize and self._plate_fit:
            self._zoom_plate()
        if watched in (self._well_viewport, self.fields.viewport()) and event.type() in (QEvent.Paint, QEvent.Resize):
            self._request_visuals()
        return super().eventFilter(watched, event)

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
        if self.isVisible() and not self._cancelled.is_set():
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
                self.preview_resolution.currentData())

    def _update_visuals(self):
        """Load up to eight visible representatives or one bounded well montage."""
        if (not self.isVisible() or self._cancelled.is_set()
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
        options = self._display_kwargs()
        resolution = self.preview_resolution.currentData()
        if self.fields.isVisible():
            self._update_field_thumbnails(signature, channels, limits, options)
        if self._comparison_signature is not None:
            return
        if self._scope[1] is not None and not self._individual and page:
            if self._displayed_montage == signature or self._montage_jobs.is_busy():
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
                self._displayed_montage = signature
                self.status.setText(tr("{count} fields; {errors} unreadable previews.",
                                       count=len(page), errors=len(errors)))
            self._montage_jobs.submit(
                lambda: _overview_montage(page, channels, limits, cancelled,
                    preview_edge=min(240, resolution), **options), show_montage)
            return
        viewport = self.wells.viewport()
        ratio = device_ratio(viewport)
        from PySide6.QtGui import QIcon
        for well in tuple(self._installed_thumbnails):
            position = self._well_positions.get(well)
            item = self.wells.item(*position) if position is not None else None
            if item is None or not viewport.rect().intersects(self.wells.visualItemRect(item)):
                if item is not None:
                    item.setIcon(QIcon())
                    item.setData(Qt.UserRole + 1, None)
                self._installed_thumbnails.discard(well)
        if self._thumbnail_jobs.is_busy():
            return
        first_row = max(0, self.wells.rowAt(0))
        last_row = self.wells.rowAt(viewport.height() - 1)
        if last_row < 0:
            last_row = self.wells.rowCount() - 1
        first_column = max(0, self.wells.columnAt(0))
        last_column = self.wells.columnAt(viewport.width() - 1)
        if last_column < 0:
            last_column = self.wells.columnCount() - 1
        requests = []
        for row in range(first_row, last_row + 1):
            for column in range(first_column, last_column + 1):
                item = self.wells.item(row, column)
                if item is None or not viewport.rect().intersects(self.wells.visualItemRect(item)):
                    continue
                well = item.data(Qt.UserRole)
                field = self._well_representatives.get(well)
                if field is None:
                    continue
                representatives = tuple(self._well_fields[well][:self.representatives.value()])
                edge = min(resolution, max(64, math.ceil(max(self._plate_cell_extent or (160, 160)) * ratio)))
                key = (tuple(record.identity for record in representatives), channels, limits, ratio, edge, self._display_options_key(options))
                if item.data(Qt.UserRole + 1) == key:
                    continue
                if key in self._thumbnail_cache:
                    self._set_thumbnail(well, key, self._thumbnail_cache[key])
                    continue
                requests.append((well, key, field, representatives, edge))
                if len(requests) == 8:
                    break
            if len(requests) == 8:
                break
        if not requests:
            return
        def read_thumbnails():
            """Read at most eight detached previews per worker lifecycle."""
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
            if cancelled.is_set():
                return
            for well, key, field, pixels, error in results:
                self._thumbnail_cache[key] = pixels
                while len(self._thumbnail_cache) > 64:
                    self._thumbnail_cache.popitem(last=False)
                if (self._visual_context() == signature
                        and device_ratio(self.wells.viewport()) == ratio):
                    self._set_thumbnail(well, key, pixels)
                    if error:
                        self.status.setText(tr("Could not read {field}: {error}",
                                               field=field.caption, error=error))
        self._thumbnail_jobs.submit(read_thumbnails, done)

    def _set_thumbnail(self, well, key, pixels):
        """Attach an image to the current well cell only on the GUI thread."""
        from PySide6.QtGui import QIcon
        position = self._well_positions.get(well)
        if position is None:
            return
        item = self.wells.item(*position)
        if not self.wells.viewport().rect().intersects(self.wells.visualItemRect(item)):
            return
        item.setData(Qt.UserRole + 1, key)
        if pixels is not None:
            image = QImage(pixels.data, pixels.shape[1], pixels.shape[0],
                           pixels.strides[0], QImage.Format_RGB888).copy()
            pixmap = scaled_for(QPixmap.fromImage(image), self.wells.viewport(),
                                self.wells.iconSize())
            item.setIcon(QIcon(pixmap))
            self._installed_thumbnails.add(well)
        else:
            item.setIcon(QIcon())
            self._installed_thumbnails.discard(well)

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
        options.update(quality_edge=self.pdf_quality.currentData(), dpi=self.pdf_dpi.currentData(),
                       conditions=dict(self._conditions), annotations=self._annotation_snapshot(),
                       field_options=self._selection_options(selected))
        self._export_jobs.submit(lambda: _overview_export(
            selected, folder, channels, limits, seed, cancelled, measurements, **options),
            lambda result: self._exported(result, measurements, folder, cancelled))

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

    def _retire(self):
        """Stop owned image generations even when a page is already hidden."""
        self._options_window.close()
        self._cancel()
        self._visual_timer.stop()
        self._native_timer.stop()
        self._measurement_cancelled.set()
        for runner in (self._index_jobs, self._image_jobs, self._export_jobs,
                       self._thumbnail_jobs, self._montage_jobs, self._field_thumbnail_jobs,
                       self._metadata_jobs, self._measurement_jobs, self._native_jobs):
            runner.shutdown(timeout_ms=0)


def _overview_workspace_qss(palette, opacity):
    """Let embedded overview content inherit the owning module's backdrop."""
    return ("QWidget#FieldOverviewWorkspace, QWidget#FieldOverviewBody, "
            "QWidget#FieldOverviewMetadata { background: transparent; }")


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
        navigation.addWidget(self.back_button)
        navigation.addWidget(QLabel(tr("Field-of-view overview")), 1)
        layout.addLayout(navigation)
        self.stack = QStackedWidget(self)
        self.stack.currentChanged.connect(self._activate_child)
        layout.addWidget(self.stack, 1)
        register_widget_qss("FieldOverviewWorkspace", _overview_workspace_qss, replace=True)
        ensure_widget_qss_applied("FieldOverviewWorkspace", root=self)

    def show_page(self):
        """Reuse module panels, falling back to the shared page host for bare callers."""
        from ..screens.map_barcodes import show_as_page
        if self._mount_module():
            self._owner.show()
            self.show()
            return
        if self._owner.layout() is None:
            layout = QVBoxLayout(self._owner)
            layout.addWidget(QWidget(self._owner), 1)
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
            figures.show()
        for splitter, pane, _collapsed in self._module_panes:
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
        self._return_child_settings()
        scroll = current._controls_scroll if isinstance(current, _FieldOverview) else current._settings_scroll
        content = scroll.takeWidget()
        clear_container_surfaces(current)
        clear_container_surfaces(content)
        clear_container_surfaces(self)
        make_transparent(self, self.stack, self._module_stack, current, content, current._module_actions)
        for label in content.findChildren(QLabel):
            label.setWordWrap(True)
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
        if self._module_figures is not None:
            height = max(700, min(32, current.wells.rowCount()) * 26 + 230) if isinstance(current, _FieldOverview) else 600
            self.setMinimumHeight(height)
            figures = self._module_figures[0]
            figures.setMinimumHeight(height + 45)
            pane = self._owner._runtime_splitter.pane("Figures")
            pane.minimum = height + 45
            pane.extent = max(pane.extent, height + 45)
        self._active_child = current

    def _leave_page(self):
        """Restore the exact host settings, image and action widgets on return."""
        from ..screens.map_barcodes import hide_as_page
        if self._module_mounted:
            self._return_child_settings()
            self._owner._settings_scroll.setWidget(self._saved_settings)
            self._saved_settings = None
            self._module_stack.setCurrentWidget(self._previous_module_image)
            self._module_stack.removeWidget(self)
            self._module_mounted = False
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


def _overview_source_path(source):
    """Normalize the first source lexically without filesystem work on the GUI thread."""
    while isinstance(source, (list, tuple)):
        source = source[0] if source else ""
    return os.path.abspath(os.path.expanduser(str(source))) if source else ""


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


def _overview_matched_fields(fields, pattern):
    """Exclude filename nonmatches while retaining invalid matched well identities."""
    import re
    if not pattern:
        return tuple(fields)
    expression = re.compile(pattern, re.I)
    return tuple(field for field in fields if all(
        expression.match(Path(path).name) for _, path in field.files))


def _overview_wizard_probe(source, pattern, convention, cancelled, source_scope="folder"):
    """Index names on a worker and inspect only the first field's image headers."""
    fields = _overview_index(source, pattern, convention, cancelled, source_scope=source_scope)
    if cancelled.is_set() or not fields:
        return fields, None
    detected_pattern, detected_label = pattern, convention
    if convention == "auto" and not pattern:
        import re
        from ..regex_detect import auto_detect_regex
        names = tuple(Path(path).name for field in fields for _, path in field.files)[:20]
        detected_pattern, detected_label, hits = auto_detect_regex(names)
        if detected_pattern and ("wellID" not in re.compile(detected_pattern).groupindex
                or not any(re.match(detected_pattern, name, re.I) for name in names)):
            detected_pattern, detected_label = pattern, convention
        if detected_pattern and hits:
            inferred = _overview_index(source, detected_pattern, "auto", cancelled,
                                       source_scope=source_scope)
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
    matched = _overview_matched_fields(fields, detected_pattern)
    first = next((field for field in matched if field.well and not field.unresolved),
                 next(iter(matched), fields[0]))
    info = dict(_overview_channel_info(first))
    info["detected_pattern"] = detected_pattern or ""
    info["detected_label"] = detected_label
    return fields, info


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
        split.addWidget(scroll)
        preview_body = QWidget()
        preview_column = QVBoxLayout(preview_body)
        split.addWidget(preview_body)
        split.setStretchFactor(1, 2)
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
        form.addRow(tr("Image source"), source_row)
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
        self.mean_match = QCheckBox(tr("Mean-matched display scaling before reference bounds"))
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
        from PySide6.QtCore import QRect
        from PySide6.QtWidgets import QStyle, QStyleOptionButton
        from ..gui_scale import _set_actual_dim

        for button in self._action_buttons:
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
        self._runner.submit(lambda: (generation, _overview_wizard_probe(
            source, pattern, "auto" if convention == "custom" else convention, cancelled, source_scope)), self._scanned)

    def _scanned(self, payload):
        """Expose actual parsed identities, available channels and embedded axes."""
        from .sortable_table import table_item
        from PySide6.QtGui import QColor
        from ..theme import active_palette
        generation, (fields, info) = payload
        if generation != self._generation or self._cancelled.is_set():
            return
        self._fields, self._info = fields, info
        if not fields or not info or not info["channels"]:
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
        self._fields = _overview_matched_fields(fields, self.regex.text())
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
        for row, (name, values) in enumerate(rows):
            from ...schema import parse_well, letters_from_row_index
            try:
                physical_row, physical_column = parse_well(values.get("wellID", ""), strict=True)
                row_letters = letters_from_row_index(int(physical_row[1:]))
                column_number = str(int(physical_column[1:]))
            except (ValueError, TypeError):
                row_letters, column_number = "", ""
            for column, value in enumerate((name, values.get("plateID", ""), values.get("wellID", ""), row_letters, column_number)
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
        self.axes.setText(tr("First field: {details}. Available channel IDs: {channels}",
                             details="; ".join(dict.fromkeys(descriptions)), channels=str(available)))
        self.time.setRange(0, max(entry.get("time_count", 1) for entry in info["files"]) - 1)
        self.z.setRange(0, max(entry.get("z_count", 1) for entry in info["files"]) - 1)
        if not self._geometry_explicit:
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
        self._readout_timer.start(0)

    def _choose_geometry(self):
        """Keep an explicit user geometry across subsequent metadata previews."""
        self._geometry_explicit = True

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
        return {"source": self.source.text(), "custom_regex": self.regex.text(),
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
