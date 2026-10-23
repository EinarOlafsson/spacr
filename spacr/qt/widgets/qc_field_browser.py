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

from ...qc_quarantine import (
    QUARANTINE_DIRNAME,
    quarantine_dir_for,
    quarantine_field,
    resolve_field_path,
    restore_field,
)
from .. import path_probe
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

    :param sink: written with ``outcome`` the instant the move has committed,
        so `QCFieldBrowser._settle_pending_move` can still report a move that
        landed while the dialog was being dismissed.
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
        #: The move in flight, or None. Holds the target it was started for,
        #: so a result cannot be applied to whatever field is on screen when
        #: it lands, and the worker's outcome, so a move that commits while
        #: the dialog is being dismissed is still announced.
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

    def _settle_pending_move(self) -> None:
        """Announce a move that committed while the dialog was dismissed.

        `JobRunner.shutdown` cancels before it drains, and a cancelled job's
        result never reaches `_apply_move`. The rename itself is NOT
        cancelled -- a stat in progress cannot be interrupted -- so without
        this a field quarantined a moment before the dialog was closed is
        moved on disk and the Measure scorecard is never told.

        Best effort by construction: dismissing with Escape never drains
        anything (`QDialog.done` deletes the dialog outright), so a move
        still running at that moment is announced by nobody. What is
        recoverable is a move that had already committed.
        """
        sink = self._pending_move
        self._pending_move = None
        if sink is None or "outcome" not in sink:
            return
        changed, _destination = sink["outcome"]
        target = sink["target"]
        active_path, quarantined_path = self._paths_for(target)
        path_probe.prime(active_path, not changed)
        path_probe.prime(quarantined_path, changed)
        self.quarantineChanged.emit(target.field, changed)

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

    @property
    def identity(self):
        """Keep the original paths in the identity, including unresolved fields."""
        return (self.plate, self.well, self.field, self.files)

    @property
    def caption(self):
        """Trace a view back to its plate, well, field and source files."""
        return " / ".join((self.plate, self.well, self.field))


def _overview_index(source, pattern="", convention="auto", cancelled=None):
    """Index paths without decoding pixels; never invent an unresolved well."""
    import re
    from ...schema import parse_well, well_id
    from ...regex_infer import _metadata_pattern
    from ..folder_metadata import _classify, _FIELD_RX, _CHANNEL_RX

    root = Path(source).resolve()
    if not root.is_dir():
        raise ValueError(tr("Choose an existing image folder."))
    custom = re.compile(pattern, re.I) if pattern else None
    groups = {}
    for directory, folders, names in os.walk(root):
        folders[:] = sorted(p for p in folders if not p.startswith("."))
        for name in sorted(names):
            if cancelled is not None and cancelled.is_set():
                return ()
            path = Path(directory) / name
            if path.suffix.lower() not in (".tif", ".tiff", ".npy", ".png", ".jpg", ".jpeg"):
                continue
            values = {}
            if custom is not None:
                match = custom.search(name)
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
            unresolved = not well or not field
            if unresolved:
                key = (plate, "", str(path.relative_to(root)), "", "")
            else:
                key = (plate, well, field, values.get("timeID", ""), values.get("sliceID", ""))
            channel = str(values.get("chanID") or "")
            if channel.isdigit():
                channel = str(int(channel))
            groups.setdefault(key, [row, column, unresolved, []])[3].append(
                (channel, str(path)))
    result = []
    for (plate, well, field, time, plane), (row, column, unresolved, files) in sorted(groups.items()):
        if time or plane:
            field += f" / T{time} / Z{plane}"
        result.append(_OverviewField(plate, well, row, column, field,
                                     tuple(sorted(files)), unresolved))
    return tuple(result)


def _overview_pixels(field, channels=(0,), percentiles=(1., 99.), edge=1200):
    """Decode one bounded field and return the same RGB pixels used in PDFs."""
    from PIL import Image, ImageDraw

    planes = []
    columns = math.ceil(math.sqrt(len(channels))) if len(channels) > 3 else 1
    tile = max(1, edge // columns)
    header = min(16, tile // 4) if len(channels) > 3 else 0
    plane_edge = max(1, tile - header)
    decoded_path, decoded_array, decoded_axes = None, None, None
    for channel in channels:
        if channel < 0:
            raise ValueError(tr("Channel indices must be zero or greater."))
        separate = any(key for key, _ in field.files)
        if separate:
            matches = [path for key, path in field.files if key == str(channel)]
            if len(matches) != 1:
                raise ValueError(tr("Channel {channel} is missing or ambiguous.", channel=channel))
            path = matches[0]
        else:
            if len(field.files) != 1:
                raise ValueError(tr("This field has ambiguous image files."))
            path = field.files[0][1]
        if path != decoded_path:
            suffix = Path(path).suffix.lower()
            if suffix == ".npy":
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                axes = "YX" if array.ndim == 2 else "YXC"
            elif suffix in (".tif", ".tiff"):
                import tifffile
                with tifffile.TiffFile(path) as tif:
                    series = tif.series[0]
                    if math.prod(series.shape) * np.dtype(series.dtype).itemsize > 128 * 1024 ** 2:
                        raise ValueError(tr("This image exceeds the 128 MiB preview decode limit."))
                    axes = series.axes
                    array = series.asarray()
            else:
                with Image.open(path) as image:
                    if image.width * image.height > 32 * 1024 ** 2:
                        raise ValueError(tr("This image exceeds the preview decode limit."))
                    array = np.asarray(image)
                    axes = "YX" if array.ndim == 2 else "YXC"
            decoded_path, decoded_array, decoded_axes = path, array, axes
        array, axes = decoded_array, decoded_axes
        if array.ndim > 3 or array.ndim < 2:
            raise ValueError(tr("Choose a two-dimensional field or a channel image stack."))
        if array.ndim == 3:
            if axes[-1] in ("C", "S"):
                array = array[..., 0 if separate else channel]
            elif axes[0] in ("C", "S", "Q", "I"):
                array = array[0 if separate else channel]
            else:
                raise ValueError(tr("The image channel axis is unresolved."))
        elif not separate and channel != 0:
            raise ValueError(tr("Channel {channel} is missing or ambiguous.", channel=channel))
        stride = max(1, math.ceil(max(array.shape) / plane_edge))
        plane = np.array(array[::stride, ::stride], dtype=np.float32, copy=True)
        finite = np.isfinite(plane)
        if not finite.any():
            raise ValueError(tr("This channel has no finite pixels."))
        if percentiles is None:
            low, high = float(plane[finite].min()), float(plane[finite].max())
        else:
            low, high = np.percentile(plane[finite], percentiles)
        plane = np.nan_to_num((plane - low) / max(float(high - low), np.finfo(np.float32).eps),
                              nan=0., posinf=1., neginf=0.)
        planes.append(np.uint8(np.clip(plane, 0, 1) * 255))
    if not planes or any(plane.shape != planes[0].shape for plane in planes):
        raise ValueError(tr("Selected channels must have matching image sizes."))
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


def _overview_export(fields, folder, channels, percentiles, seed=0, cancelled=None):
    """Write twelve bounded images per styled PDF with reproducible provenance."""
    import json
    from matplotlib.figure import Figure
    from ...plot import save_figure
    from ...figures.style import _apply_user_style

    destination = Path(folder)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.glob("overview_*.pdf")) or (destination / "overview_selection.json").exists():
        raise ValueError(tr("Overview files already exist here. Choose a new output folder."))
    written, errors, provenance = [], [], []
    for start in range(0, len(fields), 12):
        if cancelled is not None and cancelled.is_set():
            break
        figure = Figure(figsize=(12, 12))
        page_records, page_errors = [], []
        interrupted = False
        for index, field in enumerate(fields[start:start + 12]):
            if cancelled is not None and cancelled.is_set():
                interrupted = True
                break
            axis = figure.add_subplot(4, 3, index + 1)
            try:
                axis.imshow(_overview_pixels(field, channels, percentiles, edge=1000))
            except Exception as exc:
                axis.text(.5, .5, str(exc), ha="center", va="center", wrap=True)
                page_errors.append((field.caption, str(exc)))
            axis.set_title(field.caption, fontsize=8, wrap=True)
            axis.set_axis_off()
            page_records.append({"plate": field.plate, "well": field.well,
                               "field": field.field, "files": field.files})
        if interrupted:
            figure.clear()
            break
        figure.suptitle(tr("Channels: {channels}; percentiles: {percentiles}; seed: {seed}",
                          channels=str(channels), percentiles=str(percentiles), seed=seed))
        figure.tight_layout()
        _apply_user_style(figure)
        path = destination / f"overview_{start // 12 + 1:04d}.pdf"
        save_figure(figure, str(path), fmt="pdf", close=True)
        written.append(str(path))
        provenance.extend(page_records)
        errors.extend(page_errors)
    (destination / "overview_selection.json").write_text(json.dumps(
        {"channels": channels, "percentiles": percentiles, "seed": seed,
         "fields": provenance, "errors": errors,
         "cancelled": bool(cancelled is not None and cancelled.is_set()),
         "requested_field_count": len(fields), "written_field_count": len(provenance)}, indent=2), encoding="utf-8")
    return tuple(written), tuple(errors)


def _overview_montage(fields, channels, percentiles, cancelled=None):
    """Render one bounded page of field thumbnails, using the shared pixel path."""
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
            pixels = _overview_pixels(field, channels, percentiles, edge=240)
            preview = Image.fromarray(pixels)
            factor = min(240 / preview.width, 240 / preview.height)
            preview = preview.resize((max(1, round(preview.width * factor)),
                                      max(1, round(preview.height * factor))), Image.Resampling.BILINEAR)
            image.paste(preview, (x, y + 24))
        except Exception as exc:
            errors.append((field.caption, str(exc)))
            draw.text((x, y + 30), str(exc), fill="red")
    return np.asarray(image).copy(), tuple(errors)


class _OverviewWellDelegate(QStyledItemDelegate):
    """Paint each well image above its complete address and field count."""

    def paint(self, painter, option, index):
        """Use native cell geometry for the image and two-line caption."""
        from PySide6.QtCore import QRect
        from PySide6.QtGui import QPalette
        from PySide6.QtWidgets import QStyle, QStyleOptionViewItem

        opts = QStyleOptionViewItem(option)
        self.initStyleOption(opts, index)
        option.widget.style().drawPrimitive(QStyle.PE_PanelItemViewItem, opts, painter, option.widget)
        painter.save()
        rect = option.rect.adjusted(4, 4, -4, -4)
        caption_height = option.fontMetrics.height() * 2 + 4
        image_rect = QRect(rect.left(), rect.top(), rect.width(), max(0, rect.height() - caption_height))
        icon = index.data(Qt.DecorationRole)
        if icon is not None:
            icon.paint(painter, image_rect, Qt.AlignCenter)
        colour = QPalette.HighlightedText if option.state & QStyle.State_Selected else QPalette.Text
        painter.setPen(option.palette.color(colour))
        text_rect = QRect(rect.left(), rect.bottom() - caption_height + 1, rect.width(), caption_height)
        painter.drawText(text_rect, Qt.AlignCenter | Qt.TextWordWrap, str(index.data(Qt.DisplayRole) or ""))
        painter.restore()


class _FieldOverview(QDialog):
    """Shared, lazily loaded plate and field overview for both mask modules."""

    def __init__(self, parent=None, settings=None):
        """Create controls without opening a directory or decoding an image."""
        from collections import OrderedDict
        from PySide6.QtWidgets import (QLineEdit, QSpinBox, QDoubleSpinBox,
                                       QListWidget, QAbstractItemView, QTableWidget)
        import threading

        super().__init__(parent)
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
        self._image_jobs = JobRunner(self, app_key="field_overview", user_visible=False)
        self._export_jobs = JobRunner(self, app_key="field_overview")
        self._thumbnail_jobs = JobRunner(self, app_key="field_overview", user_visible=False)
        self._montage_jobs = JobRunner(self, app_key="field_overview", user_visible=False)
        self._visual_timer = QTimer(self)
        self._visual_timer.setSingleShot(True)
        self._visual_timer.timeout.connect(self._update_visuals)
        self._thumbnail_jobs.job_finished.connect(self._request_visuals)
        self._montage_jobs.job_finished.connect(self._request_visuals)
        self._scope = (None, None)
        self._image_jobs.job_finished.connect(self._refresh_image_if_pending)
        self._image_pending = False
        self._page = 0
        column = QVBoxLayout(self)
        source_row = QHBoxLayout()
        source = (settings or {}).get("src") or ""
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
        percentiles = (settings or {}).get("normalization_percentiles") or (1., 99.9)
        for spin, value in ((self.low, percentiles[0]), (self.high, percentiles[1])):
            spin.setRange(0, 100)
            spin.setValue(value)
        self.low.setToolTip(tr("Lower normalization percentile"))
        self.high.setToolTip(tr("Upper normalization percentile"))
        scan = QPushButton(tr("Load overview"))
        scan.clicked.connect(self._scan)
        metadata = QPushButton(tr("Filename metadata…"))
        metadata.clicked.connect(self._metadata)
        controls.addWidget(QLabel(tr("Channels")))
        for widget in (self.channels, self.normalise, self.low, self.high, scan, metadata):
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
        column.addLayout(navigation)
        self.wells, self.fields = QTableWidget(), QListWidget()
        self.wells.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.wells.setItemDelegate(_OverviewWellDelegate(self.wells))
        self.fields.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._well_viewport = self.wells.viewport()
        self._well_viewport.installEventFilter(self)
        self.wells.horizontalScrollBar().valueChanged.connect(self._request_visuals)
        self.wells.verticalScrollBar().valueChanged.connect(self._request_visuals)
        from PySide6.QtCore import QSize
        self.wells.setIconSize(QSize(160, 120))
        self.wells.itemDoubleClicked.connect(self._well)
        self.wells.setToolTip(tr("Select wells for export; double-click a well to explore its fields."))
        self.fields.itemSelectionChanged.connect(self._remember_selection)
        self.fields.itemClicked.connect(self._field)
        from PySide6.QtWidgets import QSplitter
        splitter = QSplitter()
        self._splitter = splitter
        splitter.addWidget(self.wells)
        splitter.addWidget(self.fields)
        self.view = ZoomableImageView()
        splitter.addWidget(self.view)
        splitter.setStretchFactor(2, 3)
        column.addWidget(splitter, 1)
        paging = QHBoxLayout()
        previous, following = QPushButton(tr("Previous fields")), QPushButton(tr("Next fields"))
        previous.clicked.connect(lambda: self._turn(-1))
        following.clicked.connect(lambda: self._turn(1))
        self.grouping = QComboBox()
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
        export, cancel = QPushButton(tr("Export PDFs")), QPushButton(tr("Cancel loading or export"))
        clear_selection = QPushButton(tr("Clear selections"))
        clear_selection.clicked.connect(self._clear_selection)
        export.clicked.connect(self._export)
        cancel.clicked.connect(self._cancel)
        for widget in (previous, following):
            paging.addWidget(widget)
        paging.addStretch(1)
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
        self.status = QLabel(tr("No images have been loaded."))
        self.status.setWordWrap(True)
        column.addWidget(self.status)
        for runner in (self._index_jobs, self._image_jobs, self._export_jobs):
            runner.job_failed.connect(self.status.setText)
        for control in (self.channels,):
            control.editingFinished.connect(self._refresh_image)
        self.normalise.toggled.connect(self._refresh_image)
        self.low.valueChanged.connect(self._refresh_image)
        self.high.valueChanged.connect(self._refresh_image)
        self._convention = str((settings or {}).get("metadata_type") or "auto")

    def _choices(self):
        """Validate one display/export configuration on the GUI thread."""
        channels = tuple(int(token.strip()) for token in self.channels.text().split(","))
        if not channels or min(channels) < 0:
            raise ValueError(tr("Choose one or more nonnegative channel indices."))
        limits = (self.low.value(), self.high.value()) if self.normalise.isChecked() else None
        if limits is not None and limits[0] >= limits[1]:
            raise ValueError(tr("The lower percentile must be below the upper percentile."))
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
        self.status.setText(tr("Cancellation requested; the current image read may finish."))
        self._index_jobs.cancel()

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
        self._fields, self._visible = (), ()
        self.wells.clear()
        self.fields.clear()
        self.view.clear_image()
        source, regex, convention, cancelled = self.source.text(), self.regex.text(), self._convention, self._cancelled
        self.status.setText(tr("Indexing image metadata…"))
        self._index_jobs.submit(lambda: _overview_index(source, regex, convention, cancelled), self._indexed)

    def _indexed(self, fields):
        """Show actual plate identities and explicitly retained unresolved fields."""
        self._fields = fields
        channel_ids = sorted({channel for field in fields for channel, _ in field.files if channel},
                             key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))
        if channel_ids and self.channels.text() == "0":
            self.channels.setText(",".join(channel_ids[:3]))
        self.plates.blockSignals(True)
        self.plates.clear()
        for plate in sorted({field.plate for field in fields}):
            self.plates.addItem(plate, plate)
        self.plates.blockSignals(False)
        self._back()
        self.status.setText(tr("{count} fields; {unresolved} with unresolved metadata.",
                               count=len(fields), unresolved=sum(f.unresolved for f in fields)))

    def _back(self):
        """Return to the current plate, retaining its identity and filters."""
        self._scope = (self.plates.currentData(), None)
        self._individual = False
        self._splitter.setSizes([720, 280, 0])
        self.view.clear_image()
        self._page = 0
        self._filter()

    def _filter(self, *_):
        """Filter metadata and show bounded field pages without image reads."""
        plate = self.plates.currentData()
        if self._scope[0] != plate:
            self._scope = (plate, None)
        available = tuple(f for f in self._fields if f.plate == plate)
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
        from PySide6.QtWidgets import QTableWidgetItem
        counts = Counter(f.well for f in available)
        addresses = {f.well: (int(f.row), f.column) for f in available if not f.unresolved}
        rows = max((position[0] for position in addresses.values()), default=0)
        columns = max((position[1] for position in addresses.values()), default=1)
        if rows > 512 or columns > 512:
            self.status.setText(tr("Plate coordinates exceed the overview grid limit."))
            return
        self._well_representatives = {}
        for field in available:
            self._well_representatives.setdefault(field.well, field)
        self._well_positions = {}
        self._installed_thumbnails.clear()
        self.wells.clear()
        self.wells.setRowCount(rows + int("" in counts))
        self.wells.setColumnCount(columns)
        self.wells.setHorizontalHeaderLabels([str(n) for n in range(1, columns + 1)])
        for well, count in sorted(counts.items()):
            r, c = addresses.get(well, (rows + 1, 1))
            item = QTableWidgetItem(tr("{well}\n{count} fields", well=well or tr("Unresolved"), count=count))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            item.setData(Qt.UserRole, well)
            self.wells.setItem(r - 1, c - 1, item)
            self._well_positions[well] = (r - 1, c - 1)
            self.wells.setRowHeight(r - 1, 150)
        for index in range(columns):
            self.wells.setColumnWidth(index, 180)
        self._visible = tuple(f for f in available if self._scope[1] is None or f.well == self._scope[1])
        self._page = min(self._page, max(0, (len(self._visible) - 1) // 24))
        self.fields.blockSignals(True)
        self.fields.clear()
        from PySide6.QtWidgets import QListWidgetItem
        for field in self._visible[self._page * 24:(self._page + 1) * 24]:
            item = QListWidgetItem(field.caption)
            item.setData(Qt.UserRole, field)
            self.fields.addItem(item)
            item.setSelected(field.identity in self._selected_fields)
        self.fields.blockSignals(False)
        self._request_visuals()

    def _remember_selection(self):
        """Retain explicit field selections across pages and plate navigation."""
        for index in range(self.fields.count()):
            item = self.fields.item(index)
            field = item.data(Qt.UserRole)
            if item.isSelected():
                self._selected_fields[field.identity] = field
            else:
                self._selected_fields.pop(field.identity, None)

    def _clear_selection(self):
        """Clear both visible and retained explicit selections."""
        self._selected_fields.clear()
        self.fields.clearSelection()
        self.wells.clearSelection()

    def _well(self, item):
        """Zoom into one well while retaining the current plate and filters."""
        self._scope = (self.plates.currentData(), item.data(Qt.UserRole))
        self._splitter.setSizes([250, 200, 750])
        self._individual = False
        self._page = 0
        self._filter()

    def _turn(self, delta):
        """Move one bounded field page at a time."""
        self._individual = False
        self._page = max(0, min((len(self._visible) - 1) // 24, self._page + delta))
        self._filter()

    def _refresh_image(self, *_):
        """Re-render only the current field after display controls change."""
        self._displayed_montage = None
        self._request_visuals()
        item = self.fields.currentItem()
        if item is not None and self._individual:
            self._field(item)

    def _field(self, item):
        """Load one selected field, with at most sixteen bounded RGB payloads cached."""
        try:
            channels, limits = self._choices()
        except (ValueError, TypeError) as exc:
            self.status.setText(str(exc))
            return
        if not self._individual:
            self._splitter.setSizes([250, 200, 750])
        self._individual = True
        field = item.data(Qt.UserRole)
        key = (field.identity, channels, limits)
        self.view.clear_image()
        if self._image_jobs.is_busy():
            self._image_pending = True
            return
        self._image_pending = False
        if key in self._cache:
            self._cache.move_to_end(key)
            self._show_pixels(self._cache[key])
            return
        cancelled = self._cancelled
        def done(pixels):
            if cancelled.is_set():
                return
            self._cache[key] = pixels
            while len(self._cache) > 16:
                self._cache.popitem(last=False)
            item = self.fields.currentItem()
            context = self._visual_context()
            if (self._individual and item is not None and item.data(Qt.UserRole).identity == field.identity
                    and context is not None and context[-2:] == (channels, limits)):
                self._show_pixels(pixels)
        self._image_jobs.submit(lambda: _overview_pixels(field, channels, limits), done)

    def eventFilter(self, watched, event):
        """Request only newly visible well images after viewport paints or resizes."""
        if watched is self._well_viewport and event.type() in (QEvent.Paint, QEvent.Resize):
            self._request_visuals()
        return super().eventFilter(watched, event)

    def showEvent(self, event):
        """Resume bounded image requests when the retained dialog is shown."""
        super().showEvent(event)
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
                self._page, channels, limits)

    def _update_visuals(self):
        """Load one visible representative or one bounded well montage at a time."""
        if not self.isVisible() or self._cancelled.is_set():
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
        if self._scope[1] is not None and not self._individual and page:
            if self._displayed_montage == signature or self._montage_jobs.is_busy():
                return
            def show_montage(result):
                pixels, errors = result
                if (cancelled.is_set() or self._visual_context() != signature or self._individual
                        or pixels is None):
                    return
                self._show_pixels(pixels)
                self._displayed_montage = signature
                self.status.setText(tr("{count} fields; {errors} unreadable previews.",
                                       count=len(page), errors=len(errors)))
            self._montage_jobs.submit(
                lambda: _overview_montage(page, channels, limits, cancelled), show_montage)
            return
        viewport = self.wells.viewport()
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
        for row in range(first_row, last_row + 1):
            for column in range(first_column, last_column + 1):
                item = self.wells.item(row, column)
                if item is None or not viewport.rect().intersects(self.wells.visualItemRect(item)):
                    continue
                well = item.data(Qt.UserRole)
                field = self._well_representatives[well]
                key = (field.identity, channels, limits)
                if item.data(Qt.UserRole + 1) == key:
                    continue
                if key in self._thumbnail_cache:
                    self._set_thumbnail(well, key, self._thumbnail_cache[key])
                    continue
                def read_thumbnail():
                    try:
                        return _overview_pixels(field, channels, limits, edge=160), ""
                    except Exception as exc:
                        return None, str(exc)
                def done(result):
                    if cancelled.is_set():
                        return
                    pixels, error = result
                    self._thumbnail_cache[key] = pixels
                    while len(self._thumbnail_cache) > 64:
                        self._thumbnail_cache.popitem(last=False)
                    if self._visual_context() == signature:
                        self._set_thumbnail(well, key, pixels)
                        if error:
                            self.status.setText(tr("Could not read {field}: {error}",
                                                   field=field.caption, error=error))
                self._thumbnail_jobs.submit(read_thumbnail, done)
                return

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
            pixmap = QPixmap.fromImage(image).scaled(
                self.wells.iconSize(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            item.setIcon(QIcon(pixmap))
            self._installed_thumbnails.add(well)
        else:
            item.setIcon(QIcon())
            self._installed_thumbnails.discard(well)

    def _refresh_image_if_pending(self, *_):
        """Coalesce rapid selections behind one active image decoder."""
        if self._image_pending:
            self._image_pending = False
            self._refresh_image()

    def _show_pixels(self, pixels):
        """Create the Qt pixmap only on the GUI thread."""
        image = QImage(pixels.data, pixels.shape[1], pixels.shape[0],
                       pixels.strides[0], QImage.Format_RGB888).copy()
        self.view.set_pixmap(QPixmap.fromImage(image))

    def _export_selection(self):
        """Export all explicit selections; sample only when selection mode is off."""
        if self.use_selection.isChecked():
            fields = tuple(self._selected_fields.values())
            if fields:
                return fields
            wells = {item.data(Qt.UserRole) for item in self.wells.selectedItems()}
            fields = tuple(f for f in self._visible if f.well in wells)
            if self.sample_selected.isChecked():
                return _overview_sample(fields, self.count.value(), self.seed.value(),
                                        self.grouping.currentData(), self.random.isChecked())
            return fields
        return _overview_sample(self._visible, self.count.value(), self.seed.value(),
                                self.grouping.currentData(), self.random.isChecked())

    def _export(self):
        """Export explicit field/well selections or reproducible examples of this view."""
        from PySide6.QtWidgets import QFileDialog
        import threading
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
        folder = QFileDialog.getExistingDirectory(self, tr("PDF output folder"))
        if not folder:
            return
        self._cancelled.set()
        self._cancelled = threading.Event()
        cancelled, seed = self._cancelled, self.seed.value()
        self._export_jobs.submit(lambda: _overview_export(selected, folder, channels, limits, seed, cancelled),
                                 self._exported)

    def _exported(self, result):
        """Report output and unreadable fields without hiding either."""
        paths, errors = result
        self.status.setText(tr("Wrote {count} PDF pages; {errors} unreadable fields.",
                               count=len(paths), errors=len(errors)))

    def closeEvent(self, event):
        """Cancel generations when closing; workers never touch the closed widgets."""
        self._cancel()
        self._visual_timer.stop()
        for runner in (self._index_jobs, self._image_jobs, self._export_jobs,
                       self._thumbnail_jobs, self._montage_jobs):
            runner.shutdown(timeout_ms=0)
        super().closeEvent(event)


def _open_field_overview(owner, settings=None):
    """Keep one shared overview window per module and retain its navigation."""
    dialog = getattr(owner, "_field_overview", None)
    if dialog is None:
        dialog = _FieldOverview(owner, settings)
        owner._field_overview = dialog
    if dialog._cancelled.is_set():
        import threading
        dialog._cancelled = threading.Event()
        dialog._image_pending = False
    dialog.show()
    dialog.raise_()
    return dialog
