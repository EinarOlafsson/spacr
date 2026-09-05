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
from PySide6.QtCore import QEvent, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGraphicsScene,
    QGraphicsView,
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


# A 4096-square camera frame is 32 MB per uint16 plane.  The browser is an
# overview, not an editor, so prepare a bounded display copy and leave the
# memory-mapped source immediately.  1600 px still resolves five-pixel QC
# objects while keeping four image planes plus masks under a modest budget.
MAX_DISPLAY_EDGE = 1600

_MASK_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "cell": (45, 220, 105),
    "nucleus": (210, 80, 255),
    "pathogen": (255, 145, 45),
    "organelle": (35, 205, 235),
}


@dataclass(frozen=True)
class QCFieldVerdict:
    """One object type's persisted verdict for the field on screen."""

    object_type: str
    severity: str
    flags: Tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class QCFieldTarget:
    """One unique plate/field target and everything QC said about it."""

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
    path = str(getattr(card, "path", "") or "")
    if path:
        return os.path.dirname(os.path.dirname(os.path.abspath(path)))
    return os.path.abspath(str(getattr(digest, "root", "") or ""))


def _field_plate(field: str) -> str:
    try:
        from ...seg_qc import parse_field_name

        return str(parse_field_name(field).plate)
    except Exception:
        return str(field).split("_", 1)[0]


def _group_digest(digest: Any) -> Dict[Tuple[str, str], Dict[str, Any]]:
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
        # A flag finding carries exact fields.  A positional finding does not:
        # every matching field is part of the pattern and belongs in the
        # browser, including individually-clean fields.
        if exact or str(getattr(finding, "kind", "")) != "clean":
            keys.append(key)
    return keys


def targets_from_digest(digest: Any) -> Tuple[QCFieldTarget, ...]:
    """Return every unique field implicated by a digest, plate-aware.

    A field flagged for cell and nucleus appears once with two verdict rows.
    Positional findings, which have no per-field flag, contribute all fields
    in their plate/object-type group so the browser does not quietly omit the
    very pattern the banner asked the user to inspect.
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
    """Return the browser targets belonging to one rendered finding."""
    groups = _group_digest(digest)
    keys = set(_keys_for_finding(finding, groups))
    return tuple(
        target for target in (
            tuple(targets) if targets is not None else targets_from_digest(digest)
        )
        if (target.plate_root, target.field) in keys
    )


def _mask_path(folder: str, field: str) -> Path:
    return Path(folder) / f"{field}.npy"


def _display_stride(shape: Sequence[int]) -> int:
    edge = max(int(shape[0]), int(shape[1]))
    return max(1, int(math.ceil(edge / float(MAX_DISPLAY_EDGE))))


def _load_mask(path: Path, stride: int, shape: Tuple[int, int]) -> np.ndarray:
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
        # Legacy arrays carry no manifest.  Their contract still appends one
        # mask plane per stack after the intensities, so the discovered stack
        # count is stronger evidence than blindly assuming four channels.
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
    # Drop the memmap before quarantine can be offered.  On Windows an open
    # mapping prevents the atomic rename; the bounded copies above own their
    # bytes independently.
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
    ``link``, an ``fsync``-ed ledger write and an ``unlink``. On the
    maintainer's sleeping ``/nas_mnt`` autofs share a single one of those had
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
    if object_type in _MASK_COLOURS:
        return _MASK_COLOURS[object_type]
    if object_type.startswith("organelle"):
        return _MASK_COLOURS["organelle"]
    # Stable and vivid for custom object-role names.
    seed = sum((index + 1) * ord(char)
               for index, char in enumerate(object_type))
    return (70 + seed % 170, 70 + (seed // 7) % 170,
            70 + (seed // 31) % 170)


def render_qc_field(
    payload: QCFieldImage,
    channel: int = -1,
    visible_masks: Iterable[str] = (),
) -> np.ndarray:
    """Render a uint8 RGB composite with object-type-coloured outlines."""
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
    height, width = rgb.shape[:2]
    image = QImage(
        rgb.data, width, height, int(rgb.strides[0]), QImage.Format_RGB888)
    return QPixmap.fromImage(image.copy())


class _FieldView(QGraphicsView):
    """Fit-on-load image canvas with wheel zoom and drag panning.

    :param parent: parent widget; ownership only.
    """

    def __init__(self, parent=None) -> None:
        """Build the view with its own scene, fitted on first load."""
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._item = None
        self._user_zoomed = False
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self._scene.clear()
        self._item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self._user_zoomed = False
        self.resetTransform()
        if not pixmap.isNull():
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def clear_image(self) -> None:
        self._scene.clear()
        self._item = None

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt override
        factor = 1.2 if event.angleDelta().y() > 0 else (1.0 / 1.2)
        self.scale(factor, factor)
        self._user_zoomed = True
        event.accept()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        if not self._user_zoomed and self._item is not None:
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)


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
        # A THIRD RUNNER, not a share of `_jobs`. The quarantine move used to
        # run inline on the GUI thread; putting it on the loading runner
        # instead would have made `_on_load_failed` report a failed rename as
        # "Could not load this field", and would have let `_show_target`'s
        # `cancel()` orphan a move mid-flight. `user_visible=True` is left at
        # its default deliberately: unlike the two above, this runner carries
        # only work the user asked for by pressing the button, and a move on a
        # slow share is exactly the kind of activity Home should own up to.
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

        self._left_shortcut = QShortcut(QKeySequence(Qt.Key_Left), self)
        self._left_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self._left_shortcut.activated.connect(self.previous_field)
        self._right_shortcut = QShortcut(QKeySequence(Qt.Key_Right), self)
        self._right_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self._right_shortcut.activated.connect(self.next_field)
        self._q_shortcut = QShortcut(QKeySequence("Q"), self)
        self._q_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self._q_shortcut.activated.connect(self.toggle_quarantine)

        self._run_timer = QTimer(self)
        self._run_timer.setInterval(400)
        self._run_timer.timeout.connect(self._sync_action)
        self._run_timer.start()
        path_probe.probes.answered.connect(self._on_probe_answered)
        # `closeEvent` is NOT a teardown hook for this dialog: it is
        # `WA_DeleteOnClose`, and Escape goes through `QDialog.done`, which
        # deletes the widget without ever raising a close event (verified on
        # Qt 6.10). `finished` is emitted on both paths, while the object is
        # still alive, which is the only place a move that committed during
        # the dismissal can still be announced.
        self.finished.connect(self._on_finished)
        self._recheck_timer = QTimer(self)
        self._recheck_timer.setInterval(2000)
        self._recheck_timer.timeout.connect(self._recheck_files)
        self._recheck_timer.start()

        # Arrow keys belong to field navigation even when a canvas, button or
        # channel picker currently owns focus.  QGraphicsView consumes arrows
        # for scrolling before a dialog-level keyPressEvent can see them, so
        # install one narrow filter across this dialog's children as well as
        # retaining QShortcuts for native shortcut dispatch.
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
        # A different field knows nothing about the last one's two copies.
        # Back to the module's own defaults until this field's load, a couple
        # of lines below, settles them exactly.
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
        self._sync_navigation()
        self._sync_action()

    def _busy(self) -> bool:
        """True while a load or a quarantine move owns this field."""
        return self._jobs.is_busy() or self._move_jobs.is_busy()

    def _sync_navigation(self) -> None:
        busy = self._busy()
        self._previous.setEnabled(not busy and self._index > 0)
        self._next.setEnabled(
            not busy and self._index + 1 < len(self._targets))

    def _draw_verdict(self, target: QCFieldTarget) -> None:
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
        self._layer_checks.clear()
        while self._layers.count():
            item = self._layers.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _on_loaded(self, payload: QCFieldImage) -> None:
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
        self._load_status.setText(tr("Could not load this field: {error}",
                                     error=message))
        self._sync_navigation()
        self._sync_action()

    def _render(self, *_args) -> None:
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
        self._view.set_pixmap(_pixmap(rgb))

    def _on_render_failed(self, message: str) -> None:
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
        """Reposition an existing browser at a banner-link target."""
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
        walk over every component, and on the maintainer's ``/nas_mnt``
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
        self._file_state()  # queues both probes again; the answers repaint

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
            # The dialog has gone; the signal outlived it.
            pass

    def _sync_action(self) -> None:
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
            # A move already running owns this field. The button is refused
            # so the same rename cannot be started twice, and the status line
            # is left exactly as it was: the move was instantaneous and
            # silent while it ran on the GUI thread, and it is not this
            # dialog's job to invent a caption for having stopped freezing.
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

        THE MOVE RUNS ON A WORKER. It did not until 2026-09-04, and that made
        the whole `path_probe` gate in `_sync_action` pointless: the poll in
        front of this button stopped freezing and the button itself still
        did, because `quarantine_dir_for` is a `Path.resolve()` realpath walk
        and the rename that follows adds a mkdir, a stat, a link, an
        ``fsync``-ed ledger write and an unlink -- on the plate path the user
        chose, which is the sleeping ``autofs`` share by assumption. See
        `_move_field`.
        """
        target = self.current_target
        if (target is None or self._is_run_active() or self._jobs.is_busy()
                or self._move_jobs.is_busy()):
            self._sync_action()
            return
        active, quarantined = self._file_state()
        if active == quarantined:  # both present or both absent
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
            # Still in flight -- repaint the button as refused. When the
            # runner is unthreaded (tests) the handlers above have already
            # run and already repainted, and syncing again here would erase
            # the failure notice one of them just wrote.
            self._sync_action()

    def _apply_move(self, sink: Dict[str, Any]) -> None:
        """Record a completed move and reload the field from its new home.

        Runs on the GUI thread, behind `JobRunner`'s generation check.
        """
        if self._pending_move is sink:
            self._pending_move = None
        target = sink["target"]
        changed, destination = sink.pop("outcome")
        # The rename just made the probe cache wrong in both directions, and
        # the truth is already in hand -- prime rather than forget, or the
        # two keys answer with the PRE-MOVE state (that is what
        # `_last_active` carries forward) until a fresh probe lands, and the
        # button offers to quarantine a file that is already in quarantine.
        active_path, quarantined_path = self._paths_for(target)
        path_probe.prime(active_path, not changed)
        path_probe.prime(quarantined_path, changed)
        self.quarantineChanged.emit(target.field, changed)
        if target is not self.current_target:
            # A banner link re-pointed the dialog while the move was in
            # flight. The move still happened and the listener above still
            # has to hear about it, but this field's notice belongs to a
            # field that is no longer on screen.
            self._sync_action()
            return
        self._action_notice = tr(
            "Quarantined {field} at {path}. Later Measure runs will skip it.",
            field=target.field, path=destination) if changed else tr(
            "Restored {field} to {path}.",
            field=target.field, path=destination)
        # Reload from the new location, both to release stale path text and to
        # prove the reversible move left a readable array.
        self._show_target(preserve_notice=True)

    def _on_move_busy_changed(self, _busy: bool) -> None:
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
        # Nothing is known about either copy after a half-finished move.
        # Forget rather than prime, and let the background probe settle it;
        # `_last_active` keeps the button showing the pre-move state in the
        # meantime, which is the state a failed move should leave.
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
        if (event.type() == QEvent.KeyPress
                and self._handle_triage_key(event.key())):
            event.accept()
            return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Keep the triage keys working when the dialog itself has focus."""
        if self._handle_triage_key(event.key()):
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
        """Retire an in-flight image load before Qt destroys the dialog."""
        self._run_timer.stop()
        self._recheck_timer.stop()
        try:
            path_probe.probes.answered.disconnect(self._on_probe_answered)
        except (RuntimeError, TypeError):
            # Escape never reaches this method at all, so the connection is
            # only ever dropped here as a courtesy; Qt has already done it
            # by the time the object is really gone.
            pass
        self._render_jobs.shutdown()
        self._jobs.shutdown()
        super().closeEvent(event)
