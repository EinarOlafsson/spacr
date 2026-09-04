"""Browse segmentation-QC fields and reversibly quarantine bad arrays.

The Measure banner already has every field verdict in its scorecards.  This
dialog turns those records into a triage loop: one implicated field at a
time, every object-type verdict together, the merged intensities under
toggleable mask outlines, and left/right/Q keyboard operation.  Loading and
render preparation run off the GUI thread; no mask is opened merely because
the Measure screen itself was shown.
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
    is_quarantined,
    quarantine_dir_for,
    quarantine_field,
    resolve_field_path,
    restore_field,
)
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
    """Fit-on-load image canvas with wheel zoom and drag panning."""

    def __init__(self, parent=None) -> None:
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
        self._jobs.cancel()
        self._render_jobs.cancel()
        language = current_language()
        self._jobs.submit(
            lambda selected=target, code=language: load_qc_field(
                selected, code),
            self._on_loaded)
        self._sync_action()

    def _on_load_busy_changed(self, _busy: bool) -> None:
        self._sync_navigation()
        self._sync_action()

    def _sync_navigation(self) -> None:
        busy = self._jobs.is_busy()
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
        if self._jobs.is_busy() or self._index <= 0:
            return
        self._index -= 1
        self._show_target()

    def next_field(self) -> None:
        """Move to the next implicated field, if one exists."""
        if self._jobs.is_busy() or self._index + 1 >= len(self._targets):
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

    def _file_state(self) -> Tuple[bool, bool]:
        target = self.current_target
        if target is None:
            return False, False
        active_path = Path(target.merged_dir, f"{target.field}.npy")
        active = active_path.is_file() and not active_path.is_symlink()
        try:
            quarantined = is_quarantined(target.merged_dir, target.field)
        except (OSError, ValueError):
            quarantined = False
        return active, quarantined

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
        """Quarantine or restore the current field; also bound to Q."""
        target = self.current_target
        if target is None or self._is_run_active() or self._jobs.is_busy():
            self._sync_action()
            return
        active, quarantined = self._file_state()
        if active == quarantined:  # both present or both absent
            self._sync_action()
            return
        try:
            if quarantined:
                destination = restore_field(
                    quarantine_dir_for(target.merged_dir), target.field)
                changed = False
                message = tr("Restored {field} to {path}.",
                             field=target.field, path=str(destination))
            else:
                destination = quarantine_field(
                    target.merged_dir, target.field,
                    flags=target.audit_flags)
                changed = True
                message = tr(
                    "Quarantined {field} at {path}. Later Measure runs will "
                    "skip it.", field=target.field, path=str(destination))
        except Exception as exc:
            LOG.exception("could not change field quarantine")
            self._sync_action()
            self._action_notice = tr(
                "Could not change quarantine: {error}", error=str(exc))
            self._action_status.setText(self._action_notice)
            return
        self._action_notice = message
        self.quarantineChanged.emit(target.field, changed)
        # Reload from the new location, both to release stale path text and to
        # prove the reversible move left a readable array.
        self._show_target(preserve_notice=True)

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

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Retire an in-flight image load before Qt destroys the dialog."""
        self._run_timer.stop()
        self._render_jobs.shutdown()
        self._jobs.shutdown()
        super().closeEvent(event)
