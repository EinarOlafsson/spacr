"""Interactive Measure crop preview with a pipeline-compatible settings panel.

The compact card is intentionally image-first.  Its ``Crop settings…`` dialog
contains the Measure settings that can be evaluated on one merged array:
general mask/channel controls, object-crop output controls, measurement
filters, and preview-only display controls.  With propagation enabled, every
pipeline setting is copied to the main Measure form as it changes.

Cell crops are grouped by three independent companion-object dimensions:
nucleated/unnucleated, infected/uninfected, and with/without organelles.  This
keeps all cells visible while making it explicit which categories would be
retained by the current filter settings.
"""
from __future__ import annotations

import itertools
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .preview_controls import (
    DEFAULT_MAX_SETS, MAX_SETS_TOOLTIP, FlatButton, FlatComboBox, FlatSpinBox,
    ImageSetSampler, apply_sample_to_combo, enumerate_image_sets,
    populate_channel_combo, selected_channel,
)
from .preview_contract import (
    PREVIEW_CANCEL_TEXT, PREVIEW_RUN_TEXT, LivePreviewContract,
)
from .channel_mapping import ChannelMappingWidget
from .percentile_pair import DECIMALS as PERCENTILE_DECIMALS
from .toggle import Toggle
from ..hidpi import logical_size, scaled_for
from ..job_runner import JobRunner
from ...crops import DEFAULT_MASK_DIMS
from ...object_roles import ALL_ROLES, ORGANELLE_ROLES, organelle_label
from ...organelle_types import (
    DEFAULT_NUMBER_OF_ORGANELLES, MAX_ORGANELLES, declared_organelle_roles,
    organelle_count, organelle_number, organelle_role_of, organelle_roles,
)

LOG = logging.getLogger("spacr.qt.measure_preview")

#: The organelle slots, as a set to test membership against. There are 702 of
#: them because `MAX_ORGANELLES` is 702, and this panel used to build a
#: mask-slice spin box, a crop-mode toggle and a minimum-area spin box for
#: every one of them -- 2,117 controls and 4,113 QWidgets against 63 settings
#: rows -- for a default `number_of_organelles` of ZERO.
_ORGANELLE_SLOTS = frozenset(ORGANELLE_ROLES)

#: The roles that are not slots, split either side of the organelle block so
#: that the order `ALL_ROLES` declares survives a run with any number of
#: slots: cell, nucleus, pathogen, <slots>, cytoplasm. Derived rather than
#: spelled out, so a role added to the schema reaches this panel too.
_BEFORE_THE_SLOTS: Tuple[str, ...] = tuple(itertools.takewhile(
    lambda role: role not in _ORGANELLE_SLOTS, ALL_ROLES))
_AFTER_THE_SLOTS: Tuple[str, ...] = tuple(
    role for role in ALL_ROLES[len(_BEFORE_THE_SLOTS):]
    if role not in _ORGANELLE_SLOTS)

#: Mask-slice defaults for the fixed roles. The slots are left out because
#: every slot control starts at -1 ("Not present") whatever
#: :data:`DEFAULT_MASK_DIMS` says -- which is what building from the whole
#: dict did as well, only 702 times over.
_MASK_DIMS = {name: value for name, value in DEFAULT_MASK_DIMS.items()
              if name not in _ORGANELLE_SLOTS}
_SUPPORTED = (".npy",)

#: Where an organelle slot control starts out. See :data:`_MASK_DIMS`.
_SLOT_MASK_DIM = -1


def _objects_for(count: int) -> Tuple[str, ...]:
    """The object roles a run with ``count`` organelle slots has.

    The panel's object vocabulary, in the order `ALL_ROLES` declares: the
    fixed roles with exactly as many organelle slots between them as the run
    says it has. A count of zero -- the default, and the common case --
    answers the four fixed roles alone.

    :param count: how many organelle slots the run declares.
    :returns: the roles, in display order.
    """
    return (*_BEFORE_THE_SLOTS, *organelle_roles(count), *_AFTER_THE_SLOTS)


def _slots_the_settings_speak_for(settings: Dict[str, Any]) -> int:
    """How many organelle slots a settings dict needs controls for.

    THE COUNT IS NOT THE WHOLE ANSWER. `active_organelle_roles` is what the
    panel SHOWS, but a file written at seven and opened at two still carries
    slots three to seven, and this panel writes its controls back out: a slot
    with no control propagates nothing, so lowering the count would rewrite
    the user's file. :func:`declared_organelle_roles` is the union the
    settings machinery already uses for exactly that reason.

    :param settings: the settings about to be applied.
    :returns: the number of slots to have controls for.
    """
    return len(declared_organelle_roles(settings))


def resolve_merged_source(path, rng=None):
    """A concrete merged ``.npy`` from whatever the user gave us.

    THREE THINGS ARE A VALID ANSWER TO "where are the crops", and only one of
    them used to be accepted:

    * a merged ``.npy`` -- what the Choose dialog offered, and nothing else;
    * a RUN FOLDER, which is what `src` holds. A Measure run is pointed at the
      plate directory and finds `merged/` itself, so the preview asking for a
      file meant hunting through a folder for one of fifty-two arrays whose
      names carry a well and a field and nothing about which is interesting;
    * a `merged/` folder directly.

    A field is picked at RANDOM rather than taking the first. Sorted first is
    always the same well and the same field, so a preview that always opens on
    `E01` field 1 tells the user about one corner of one condition -- and if
    that field happens to be clean, a crop size that cuts cells in half
    everywhere else looks fine.

    No Qt and no widgets: this runs on the worker with the load it feeds.

    :param path: a file, a run folder, or a merged folder.
    :param rng: something with ``choice``; defaults to :mod:`random`.
    :returns: a :class:`Path` to one array, or ``None``.
    """
    import random as _random

    if not path:
        return None
    candidate = Path(str(path).strip())
    if candidate.is_file():
        return candidate
    if not candidate.is_dir():
        return None
    for folder in (candidate / "merged", candidate):
        if not folder.is_dir():
            continue
        arrays = sorted(folder.glob("*.npy"))
        if arrays:
            return (rng or _random).choice(arrays)
    return None


def load_merged_array(path: str, enumerate_sets: bool = True
                      ) -> Dict[str, Any]:
    """Read one merged ``(H, W, C)`` array. No Qt, so it runs on a worker.

    Also lists the sibling arrays, by name only, for the same reason
    ``live_preview.load_source_payload`` does: ``_refresh_source_selectors``
    enumerates the folder on every load, and doing that on the GUI thread cost
    124 ms of the 2469 ms freeze this replaced.

    :param path: NumPy array file to open; a usable payload must decode to one
        merged ``(height, width, channels)`` array.
    :param enumerate_sets: ``False`` reuses the sampler's cached listing --
        the FOV dropdown hands out a path it already enumerated.
    :returns: ``{path, data, directory, sets, channels, error}``. ``data`` is
        ``None`` whenever ``error`` is set or the file is not a merged array.
    """
    out: Dict[str, Any] = {"path": path, "data": None, "directory": None,
                           "sets": None, "channels": None, "error": ""}
    resolved = resolve_merged_source(path)
    if resolved is None:
        out["error"] = (f"No merged .npy found at {path}"
                        if Path(str(path)).is_dir()
                        else f"Not a file or folder: {path}")
        return out
    path = str(resolved)
    out["path"] = path
    if enumerate_sets:
        try:
            sets, channels = enumerate_image_sets(Path(path).parent, _SUPPORTED)
            out["directory"] = str(Path(path).parent)
            out["sets"] = sets
            out["channels"] = channels
        except Exception:
            LOG.exception("Could not enumerate merged arrays beside %s", path)
    try:
        data = np.load(path)
    except Exception as exc:
        out["error"] = f"Failed to load: {exc}"
        return out
    if data.ndim != 3:
        out["error"] = (
            f"Expected a merged (H,W,C) array; got shape {data.shape}")
        return out
    out["data"] = data
    return out


def _presence_in(data: np.ndarray, dim: Optional[int],
                 cell_region: np.ndarray, minimum: int) -> Optional[bool]:
    """Is a companion object present inside ``cell_region``?

    Split out of ``MeasurePreviewPanel._presence`` so the scan -- which is
    ``O(crops x labels x H x W)`` and was the slowest thing on the GUI thread
    after the read itself -- can run on a worker. Reads no widget: every
    threshold arrives as an argument.
    """
    if dim is None or dim >= data.shape[2]:
        return None
    mask = data[..., dim].astype(np.int64, copy=False)
    labels = np.unique(mask[cell_region])
    labels = labels[labels > 0]
    if minimum <= 0:
        return bool(labels.size)
    for label in labels:
        if int(np.count_nonzero(mask == label)) >= minimum:
            return True
    return False


def _phenotype_label(name: str, value: Optional[bool]) -> str:
    """Render one phenotype as the word a biologist would use.

    ``Nucleus``/``True`` is "Nucleated" rather than "Nucleus: yes" --
    the reader is scanning crops, not reading a table.

    :param name: the compartment.
    :param value: whether it is present; ``None`` renders as not applicable,
        which is different from absent.
    :returns: the label.
    """
    if value is None:
        return f"{name} n/a"
    if name == "Nucleus":
        return "Nucleated" if value else "Unnucleated"
    if name == "Pathogen":
        return "Infected" if value else "Uninfected"
    return "Organelle+" if value else "Organelle−"


def annotate_crops(crops: List[Dict[str, Any]], data: Optional[np.ndarray],
                   params: Dict[str, Any]) -> None:
    """Tag each crop with its phenotype category and whether filters keep it.

    ``params`` is a snapshot of the widget values taken on the GUI thread --
    see :meth:`MeasurePreviewPanel._category_params`. Passing a snapshot rather
    than reading the widgets is what makes this safe to call from a worker.
    """
    object_name = params.get("object", "cell")
    if data is None or object_name != "cell":
        for entry in crops:
            entry["category"] = object_name.capitalize()
            entry["included"] = True
        return
    cell_dim = params.get("cell_dim")
    if cell_dim is None or cell_dim >= data.shape[2]:
        return
    dims = params.get("dims", {})
    minima = params.get("minima", {})
    allow_uninfected = bool(params.get("uninfected", False))
    cell_mask = data[..., cell_dim].astype(np.int64, copy=False)
    for entry in crops:
        region = cell_mask == int(entry["label"])
        nucleus = _presence_in(data, dims.get("nucleus"), region,
                               int(minima.get("nucleus", 0)))
        pathogen = _presence_in(data, dims.get("pathogen"), region,
                                int(minima.get("pathogen", 0)))
        organelle = _presence_in(data, dims.get("organelle"), region,
                                 int(minima.get("organelle", 0)))
        entry["phenotype"] = {
            "nucleus": nucleus,
            "pathogen": pathogen,
            "organelle": organelle,
        }
        entry["category"] = " · ".join((
            _phenotype_label("Nucleus", nucleus),
            _phenotype_label("Pathogen", pathogen),
            _phenotype_label("Organelle", organelle),
        ))
        included = nucleus is not False
        if not allow_uninfected:
            included = included and pathogen is True
        entry["included"] = bool(included)


def compute_crops(data: np.ndarray, crop_kwargs: Dict[str, Any],
                  category_params: Dict[str, Any]) -> Dict[str, Any]:
    """Crop the objects out of ``data`` and categorise them. Worker-safe.

    The whole of what ``refresh`` used to do inline, minus the drawing:
    ``QPixmap`` is a GUI object and building one off the GUI thread is
    undefined behaviour, so the pixmaps stay in :meth:`_render_grid`.

    :returns: ``{crops, error}``.
    """
    from spacr.measure import crop_objects_from_array

    try:
        crops = crop_objects_from_array(data, **crop_kwargs)
    except Exception as exc:
        return {"crops": [], "error": f"Crop failed: {exc}"}
    annotate_crops(crops, data, category_params)
    return {"crops": crops, "error": ""}


def _rounded_pixmap(pm: QPixmap, radius: int = 8) -> QPixmap:
    """``pm`` with its corners rounded, at the density it was drawn at.

    The canvas takes the source's device pixel ratio, so the rounding is
    done in the same LOGICAL coordinates the picture is laid out in. Left at
    1.0 it would paint a dense thumbnail at half size into the corner of a
    box twice as large, which is a quarter-size crop on any HiDPI screen.
    ``radius`` is 8 logical px on every display, which is the point of
    rounding a corner rather than counting pixels into it.
    """
    if pm.isNull():
        return pm
    out = QPixmap(pm.size())
    out.setDevicePixelRatio(pm.devicePixelRatio())
    out.fill(Qt.transparent)
    painter = QPainter(out)
    painter.setRenderHint(QPainter.Antialiasing, True)
    shown = logical_size(pm)
    path = QPainterPath()
    path.addRoundedRect(QRectF(0, 0, shown.width(), shown.height()),
                        radius, radius)
    painter.setClipPath(path)
    painter.drawPixmap(0, 0, pm)
    painter.end()
    return out


def _parse_channels(text: str) -> List[int]:
    """Parse a channel list from typed text.

    Semicolons are accepted as separators alongside commas, and anything
    that is not a plain number is dropped -- so a half-typed entry narrows
    the preview rather than emptying it.

    :param text: the typed list.
    :returns: the channel indices, in the order given.
    """
    out = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
    return out


def _optional_spin_value(widget: QSpinBox) -> Optional[int]:
    """Read a spin box whose negative range means "unset".

    :param widget: the spin box.
    :returns: the value, or ``None`` when it is negative -- which is how a
        spin box says "no limit" without a second control beside it.
    """
    value = int(widget.value())
    return None if value < 0 else value


def _default_png_mapping() -> Dict[str, Optional[int]]:
    """The run's own default crop colouring, read rather than copied.

    Imported inside the call for the reason ``compute_crops`` does the same
    with ``spacr.measure``: this module is imported to build a screen, and a
    constant is not worth 77 ms of import at that moment. A literal copy
    would be free and is exactly how the preview and the run came to
    disagree in the first place.
    """
    try:
        from spacr.crops import DEFAULT_PNG_CHANNEL_MAPPING
        return dict(DEFAULT_PNG_CHANNEL_MAPPING)
    except Exception:      # pragma: no cover - crops is a hard dependency
        LOG.debug("could not read the default png mapping", exc_info=True)
        return {"r": 2, "g": 1, "b": 0}


def _resolve_png_mapping(settings) -> Dict[str, Optional[int]]:
    """``png_channel_mapping``, or the legacy ``png_dims``, or the default.

    The run's own precedence, reached through the run's own function, so a
    settings dict seeds this panel with the colours it will actually get.
    """
    try:
        from spacr.crops import resolve_png_channel_mapping
        return resolve_png_channel_mapping(settings)
    except Exception:
        LOG.debug("could not resolve the png mapping", exc_info=True)
        return _default_png_mapping()


def _mapping_to_rgb_list(mapping: Dict[str, Optional[int]]) -> List[int]:
    """``{r, g, b}`` -> the RGB-ordered channel list the cropper takes.

    ``crop_objects_from_array``'s ``channels`` argument is RGB order, so the
    mapping the run resolves and the list the preview draws with are the same
    thing written two ways. A colour mapped to ``None`` is an empty plane in
    the run; it is dropped here, which is the closest the three-channel
    preview grid can get.
    """
    return [int(mapping[k]) for k in ("r", "g", "b")
            if mapping.get(k) is not None]


class _CropThumb(QLabel):
    """One crop in the preview grid, with an included/excluded border.

    :param index: this crop's position in the grid, and the payload emitted
        with :attr:`clicked` -- so it is how the panel knows WHICH thumb was
        pressed, not merely where it sits.
    :param included: whether the crop is in the measurement. Drawn as the
        border colour, accent for in and dim for out; it is the only
        indication, so a thumb built with the wrong value looks like a
        correctly excluded one.
    :param parent: parent widget; ownership only.
    """

    clicked = Signal(int)

    def __init__(self, index: int, *, included: bool = True, parent=None):
        """Build the thumb, rimmed by whether the crop is included."""
        super().__init__(parent)
        self._index = index
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.PointingHandCursor)
        try:
            from ..theme import active_palette
            palette = active_palette()
            border = palette["accent"] if included else palette["fg_dim"]
            background = palette["surface_hi"]
        except Exception:
            border, background = ("#4A9EFF", "#24262a")
        self.setStyleSheet(
            "QLabel {"
            f"background: {background}; border: 2px solid {border};"
            "border-radius: 9px; padding: 2px;"
            "}"
        )

    def mousePressEvent(self, event):
        """Announce this crop's index when clicked.

        :param event: the mouse event.
        """
        self.clicked.emit(self._index)
        super().mousePressEvent(event)


class MeasurePreviewPanel(LivePreviewContract, QWidget):
    """Preview Measure crops and propagate a faithful run configuration.

    A live view like the other three, and since the shared contract
    (:class:`~spacr.qt.widgets.preview_contract.LivePreviewContract`) it
    wears their vocabulary: the same **Run preview** button, the same
    **Cancel** beside it, and a sentence on the status line whenever it
    cannot preview. It used to be alone in every one of those columns —
    its button said "Refresh crops", nothing could be cancelled, and a
    press with no array loaded did nothing and said nothing.

    :param parent: parent widget.
    :param threaded: whether the panel's jobs run off the GUI thread. False
        runs each one inline, emitting the same signals in the same order, so
        a test can drive the panel synchronously without the behaviour
        diverging.
    """

    preview_ready = Signal(object)

    PREVIEW_SOURCE_HINT = "Load a merged array first."

    def __init__(self, parent=None, *, threaded: bool = True):
        """Build the preview: its controls, its grid and its drop target.

        :param parent: parent widget.
        """
        super().__init__(parent)
        self._data: Optional[np.ndarray] = None
        #: The `src` already auto-loaded from, so a settings change
        #: that does not move `src` cannot re-randomise the field.
        self._auto_loaded_src: str = ""
        self._data_path: Optional[str] = None
        self._crops: List[Dict[str, Any]] = []
        self._selected: set[int] = set()
        self._propagate_cb = None
        self._thumb_px = 132
        self._crop_settings_dialog: Optional[CropSettingsDialog] = None
        self._jobs = JobRunner(self, threaded=threaded,
                               app_key="measure preview")
        #: Bumped whenever a load or a re-crop supersedes the one in flight.
        self._load_token = 0
        self._crop_token = 0
        self._loading_fov = False
        self._sampler = ImageSetSampler(DEFAULT_MAX_SETS)
        self._build_controls()
        self._build_ui()
        self._connect_controls()
        self.setAcceptDrops(True)
        from ..screens.settings_model import retarget_field_tooltips
        retarget_field_tooltips(self)


    def _object_names(self, count: Optional[int] = None) -> Tuple[str, ...]:
        """The objects this panel has controls for, in display order.

        :param count: how many organelle slots to name. Defaults to the slots
            already built, which is what every consumer of the control dicts
            wants; :meth:`_build_slot_controls` passes the new total.
        :returns: the role names.
        """
        if count is None:
            count = getattr(self, "_slots_built",
                            DEFAULT_NUMBER_OF_ORGANELLES)
        return _objects_for(count)

    @staticmethod
    def _in_role_order(controls: Dict[str, QWidget],
                       order: Tuple[str, ...]) -> Dict[str, QWidget]:
        """``controls`` re-keyed into ``order``.

        The control dicts are iterated to lay the dialog out, to propagate and
        to take the widget census, so their ORDER is the order the user reads.
        A slot built on demand is appended, which would put Organelle 1 after
        Cytoplasm; this puts it back where `ALL_ROLES` says it goes.

        :param controls: the per-object controls.
        :param order: the roles, in the order they should be read in.
        :returns: the same widgets, in ``order``.
        """
        return {name: controls[name] for name in order if name in controls}

    @staticmethod
    def _spin(
        lo: int,
        hi: int,
        value: int,
        *,
        special: str = "",
        parent=None,
    ) -> QSpinBox:
        """One labelled spin box, wired to the settings it edits.

        :param lo: the lowest value it accepts.
        :param hi: the highest.
        :param value: where it starts.
        :param special: text shown in place of the minimum, if any.
        :param parent: parent widget.
        :returns: the spin box.
        """
        widget = QSpinBox(parent)
        widget.setRange(lo, hi)
        widget.setValue(value)
        if special:
            widget.setSpecialValueText(special)
        return widget

    def _build_controls(self) -> None:
        """Build the control row: object, crop modes and sizes.

        FOR THE OBJECTS THE RUN HAS, not for every object spaCR can name.
        :data:`ALL_ROLES` carries 702 organelle slots and this built three
        controls for each of them, so opening Measure constructed ~2,117
        controls -- 4,113 QWidgets, each spin box dragging a QLineEdit and a
        validator and each toggle a QPropertyAnimation -- against 63 settings
        rows, while the default `number_of_organelles` is ZERO. The slots a
        run declares arrive later through :meth:`set_organelle_count`, which
        builds and wires them then.
        """
        #: The slots this panel has controls for. Grows with the declared
        #: count and never shrinks: lowering the count HIDES its rows, and
        #: the controls keep their answers so raising it again brings them
        #: back rather than a row of defaults.
        self._slots_built = DEFAULT_NUMBER_OF_ORGANELLES
        self._organelle_count = DEFAULT_NUMBER_OF_ORGANELLES
        self._experiment = QLineEdit("experiment", self)
        self._measurement_channels = QLineEdit("0,1,2,3", self)
        self._object_box = QComboBox(self)
        self._object_box.addItems(self._object_names())
        self._mask_dims = {
            name: self._spin(-1, 64, value, special="Not present", parent=self)
            for name, value in _MASK_DIMS.items()
        }
        self._cytoplasm = Toggle(parent=self)
        self._plot = Toggle(parent=self)
        self._test_mode = Toggle(parent=self)
        self._timelapse = Toggle(parent=self)

        self._save_png = Toggle(parent=self)
        self._save_png.setChecked(True)
        self._save_arrays = Toggle(parent=self)
        self._crop_mode_checks = {
            name: Toggle(name.capitalize(), self)
            for name in self._object_names()
        }
        self._crop_mode_checks["cell"].setChecked(True)
        self._crop_size = self._spin(16, 2048, 224, parent=self)
        self._crop_width = self._crop_size
        self._crop_height = self._crop_size
        self._non_square_png_size = None
        #: Waits out the digits of a typed crop size before re-cutting.
        self._crop_size_timer = QTimer(self)
        self._crop_size_timer.setSingleShot(True)
        self._crop_size_timer.setInterval(250)
        self._crop_size_timer.timeout.connect(self.refresh)
        self._png_dims = ChannelMappingWidget(_default_png_mapping(), self)
        self._use_bbox = Toggle(parent=self)
        self._buffer = self._spin(0, 200, 10, parent=self)
        self._normalise = Toggle(parent=self)
        self._normalise.setChecked(True)
        self._lo_pct = QDoubleSpinBox(self)
        self._lo_pct.setDecimals(PERCENTILE_DECIMALS)
        self._lo_pct.setRange(0.0, 50.0)
        self._lo_pct.setSingleStep(0.01)
        self._lo_pct.setValue(1.0)
        self._lo_pct.setSuffix(" %")
        self._hi_pct = QDoubleSpinBox(self)
        self._hi_pct.setDecimals(PERCENTILE_DECIMALS)
        self._hi_pct.setRange(50.0, 100.0)
        self._hi_pct.setSingleStep(0.01)
        self._hi_pct.setValue(99.0)
        self._hi_pct.setSuffix(" %")
        self._normalize_by = QComboBox(self)
        self._normalize_by.addItems(("png", "fov"))
        self._dilate = Toggle(parent=self)
        self._dilate_ratio = QDoubleSpinBox(self)
        self._dilate_ratio.setRange(0.0, 10.0)
        self._dilate_ratio.setSingleStep(0.05)
        self._dilate_ratio.setValue(0.2)

        self._min_sizes = {
            name: self._spin(0, 10_000_000, 0, parent=self)
            for name in self._object_names()
        }
        self._uninfected = Toggle(parent=self)
        self._uninfected.setChecked(True)
        self._merge_edge_pathogen_cells = Toggle(parent=self)
        self._merge_edge_pathogen_cells.setChecked(True)

        self._max_area = self._spin(0, 100_000_000, 0, parent=self)
        self._max_crops = self._spin(1, 1000, 60, parent=self)
        self._group_cells = Toggle(parent=self)
        self._group_cells.setChecked(True)
        self._propagate_btn = QPushButton("Propagate settings", self)
        self._propagate_btn.setObjectName("ToggleButton")
        self._propagate_btn.setCheckable(True)
        self._propagate_btn.setToolTip(
            "When on, changes made here are copied into the main Measure "
            "settings."
        )

        self._mask_dim = self._mask_dims["cell"]
        self._min_area = self._min_sizes["cell"]
        self._channels = self._png_dims

        tooltips = {
            self._measurement_channels:
                "Image channels measured from the merged array.",
            self._png_dims:
                "Image channel indices written to crop R, G and B planes.",
            self._use_bbox:
                "Keep the padded rectangular bounding box instead of masking "
                "pixels outside the object.",
            self._normalise:
                "Write False when off or [lower, upper] percentiles when on.",
            self._uninfected:
                "Keep uninfected cells. Off marks them as excluded.",
            self._group_cells:
                "Group cells by nucleus, pathogen and organelle presence.",
        }
        for widget, text in tooltips.items():
            widget.setToolTip(text)
        for widget in self._managed_widgets():
            widget.hide()

    def _build_ui(self) -> None:
        """Lay out the controls over the thumbnail grid."""
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        pick_row = QHBoxLayout()
        self._pick_row = pick_row
        self._path_label = QLabel(
            "No array loaded — drop a merged .npy here, or choose one")
        self._path_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._max_sets_box = FlatSpinBox(self, value=DEFAULT_MAX_SETS,
                                         tooltip=MAX_SETS_TOOLTIP)
        self._max_sets_box.valueChanged.connect(self._on_max_sets_changed)
        self._fov_box = FlatComboBox(
            self,
            tooltip=("Field of view. Lists a random sample of the merged .npy "
                     "arrays beside the loaded one; picking one loads it."))
        self._fov_box.currentIndexChanged.connect(self._on_fov_changed)
        self._channel_box = FlatComboBox(
            self,
            tooltip=("Displayed channel. 'All channels' renders the crops "
                     "from the PNG channels in Crop settings; picking one "
                     "shows that channel alone."))
        self._channel_box.currentIndexChanged.connect(
            self._on_display_channel_changed)
        populate_channel_combo(self._channel_box, 0)
        self._pick_btn = FlatButton("Choose merged array…", self)
        self._pick_btn.clicked.connect(self._pick_file)
        self._paste_box = QLineEdit(self)
        self._paste_box.setPlaceholderText("…or paste a path")
        self._paste_box.setToolTip(
            "Paste a merged .npy, a merged folder, or a run folder (the one "
            "you would put in src). Press Enter to load a field from it.")
        self._paste_box.setClearButtonEnabled(True)
        self._paste_box.returnPressed.connect(self._load_the_pasted_path)
        pick_row.addWidget(self._path_label, 1)
        pick_row.addWidget(self._max_sets_box)
        pick_row.addWidget(self._fov_box)
        pick_row.addWidget(self._channel_box)
        pick_row.addWidget(self._paste_box, 1)
        pick_row.addWidget(self._pick_btn)
        root.addLayout(pick_row)

        actions = QHBoxLayout()
        self._run_btn = QPushButton(PREVIEW_RUN_TEXT)
        self._run_btn.clicked.connect(self.run_preview)
        self._refresh_btn = self._run_btn
        self._cancel_btn = QPushButton(PREVIEW_CANCEL_TEXT)
        self._cancel_btn.setToolTip(
            "Abandon the crop pass in flight; its result is dropped.")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self.cancel_preview)
        self._settings_btn = QPushButton("Crop settings…")
        self._settings_btn.clicked.connect(self.open_crop_settings)
        self._status = QLabel("")
        actions.addWidget(self._run_btn)
        actions.addWidget(self._cancel_btn)
        actions.addWidget(self._settings_btn)
        actions.addWidget(self._status, 1)
        from .preview_scale import install_preview_scale
        self._scale_control = install_preview_scale(self, "measure", actions)
        self._scale_control.scaler.add_hook(self._on_preview_scale)
        root.addLayout(actions)

        self._grid_scroll = QScrollArea()
        self._grid_scroll.setWidgetResizable(True)
        self._grid_scroll.setFrameShape(QScrollArea.NoFrame)
        try:
            from ..theme import active_palette
            background = active_palette()["surface_alt"]
        except Exception:
            background = "#161719"
        self._grid_scroll.viewport().setStyleSheet(
            f"background: {background};")
        self._grid_holder = QWidget()
        self._grid_holder.setObjectName("MeasureGrid")
        self._grid_holder.setStyleSheet(
            f"QWidget#MeasureGrid {{ background: {background}; }}")
        self._grid = QGridLayout(self._grid_holder)
        self._grid.setSpacing(8)
        self._grid.setContentsMargins(8, 8, 8, 8)
        self._grid_scroll.setWidget(self._grid_holder)
        root.addWidget(self._grid_scroll, 1)

    def _managed_widgets(self) -> List[QWidget]:
        """Every control this panel owns, for gating and for propagation.

        :returns: the widgets, keyed by setting name.
        """
        widgets: List[QWidget] = [
            self._experiment, self._measurement_channels, self._object_box,
            *self._mask_dims.values(), self._cytoplasm, self._plot,
            self._test_mode, self._timelapse, self._save_png,
            self._save_arrays, *self._crop_mode_checks.values(),
            self._crop_size,
            self._png_dims, self._use_bbox, self._buffer, self._normalise,
            self._lo_pct, self._hi_pct, self._normalize_by, self._dilate,
            self._dilate_ratio, *self._min_sizes.values(), self._uninfected,
            self._merge_edge_pathogen_cells, self._max_area,
            self._max_crops, self._group_cells, self._propagate_btn,
        ]
        return widgets

    #: The signal a control announces a change on, most specific first. The
    #: FIRST one it has is the one connected: a spin box has `valueChanged`
    #: and `editingFinished` both, and wiring both re-previews twice.
    _REFRESH_SIGNALS = ("valueChanged", "currentTextChanged",
                        "editingFinished", "toggled")
    #: The same, plus the one a line edit announces on.
    _PROPAGATE_SIGNALS = _REFRESH_SIGNALS + ("textChanged",)

    @staticmethod
    def _wire(widget: QWidget, signal_names: Tuple[str, ...], slot) -> bool:
        """Connect ``slot`` to the first of ``signal_names`` ``widget`` has.

        THE ONE PLACE A CONTROL IS WIRED, so that a slot control built later
        by :meth:`_build_slot_controls` is wired exactly as one built at
        startup. A lazily created widget nobody connected is a silent dead
        control: it looks right and changes nothing.

        :param widget: the control to wire.
        :param signal_names: candidate signals, most specific first.
        :param slot: what to call when it changes.
        :returns: whether anything was connected.
        """
        for signal_name in signal_names:
            signal = getattr(widget, signal_name, None)
            if signal is None:
                continue
            try:
                signal.connect(slot)
                return True
            except (TypeError, RuntimeError):
                pass
        return False

    def _wire_object_control(self, widget: QWidget, *,
                             refreshes: bool) -> None:
        """Wire one per-object control the way `_connect_controls` does.

        :param widget: the mask-slice, minimum-area or crop-mode control.
        :param refreshes: True for the controls the preview re-crops for --
            the mask slices and the size floors; False for the crop-mode
            toggles, which only propagate.
        """
        if refreshes:
            self._wire(widget, self._REFRESH_SIGNALS, self._on_setting_changed)
        else:
            self._wire(widget, self._PROPAGATE_SIGNALS, self._maybe_propagate)

    def _connect_controls(self) -> None:
        """Wire each control to the refresh it should trigger."""
        self._object_box.currentTextChanged.connect(self._on_object_changed)
        self._crop_size.valueChanged.connect(self._on_crop_size_changed)
        self._normalise.toggled.connect(self._refresh_control_gates)
        self._dilate.toggled.connect(self._refresh_control_gates)
        self._use_bbox.toggled.connect(self._refresh_control_gates)

        refresh_widgets = [
            self._object_box, *self._mask_dims.values(), self._png_dims,
            self._use_bbox, self._buffer, self._normalise, self._lo_pct,
            self._hi_pct, *self._min_sizes.values(), self._uninfected,
            self._max_area, self._max_crops, self._group_cells,
        ]
        for widget in refresh_widgets:
            self._wire(widget, self._REFRESH_SIGNALS, self._on_setting_changed)

        for widget in self._managed_widgets():
            if widget in refresh_widgets or widget is self._propagate_btn:
                continue
            self._wire(widget, self._PROPAGATE_SIGNALS, self._maybe_propagate)
        self._propagate_btn.toggled.connect(self._on_propagate_toggled)
        self._refresh_control_gates()


    def open_crop_settings(self) -> None:
        """Open the crop-settings dialog for this preview."""
        dialog = self._crop_settings_dialog
        if dialog is not None and dialog.isVisible():
            dialog.raise_()
            dialog.activateWindow()
            return
        dialog = CropSettingsDialog(self)
        self._crop_settings_dialog = dialog
        dialog.finished.connect(self._clear_crop_settings_dialog)
        dialog.show()

    def _clear_crop_settings_dialog(self, *_args) -> None:
        """Forget the crop dialog once it has closed.

        HELD ONLY WHILE OPEN, so a second press builds a fresh one rather than
        re-showing a dialog whose C++ half has gone.
        """
        self._crop_settings_dialog = None

    def _build_slot_controls(self, count) -> None:
        """Bring the controls for organelle slots 1 to ``count`` into existence.

        THE SLOTS ARE BUILT HERE AND NOWHERE ELSE, which is what keeps
        opening Measure from constructing 2,117 controls for a run that
        declares no organelle at all. Each new control is wired through
        :meth:`_wire_object_control` in the same breath as it is made: a
        control created later that nobody connected is a dead control, and
        it looks exactly like a working one.

        GROWS ONLY. Lowering the count HIDES rows -- see
        :meth:`CropSettingsDialog.refresh_organelle_slots` -- so a slot built
        once keeps its answers and raising the count again brings them back
        rather than a row of defaults. That is the settings grid's rule, cut
        from the view and not from the settings.

        :param count: how many slots to have controls for.
        """
        try:
            wanted = max(0, min(int(count), MAX_ORGANELLES))
        except (TypeError, ValueError):
            return
        if wanted <= self._slots_built:
            return
        for role in organelle_roles(wanted):
            if role in self._mask_dims:
                continue
            mask_dim = self._spin(-1, 64, _SLOT_MASK_DIM,
                                  special="Not present", parent=self)
            min_size = self._spin(0, 10_000_000, 0, parent=self)
            crop_mode = Toggle(role.capitalize(), self)
            self._mask_dims[role] = mask_dim
            self._min_sizes[role] = min_size
            self._crop_mode_checks[role] = crop_mode
            self._object_box.insertItem(
                len(_BEFORE_THE_SLOTS) + organelle_number(role) - 1, role)
            self._wire_object_control(mask_dim, refreshes=True)
            self._wire_object_control(min_size, refreshes=True)
            self._wire_object_control(crop_mode, refreshes=False)
            for widget in (mask_dim, min_size, crop_mode):
                widget.hide()
        self._slots_built = wanted
        order = self._object_names()
        self._mask_dims = self._in_role_order(self._mask_dims, order)
        self._min_sizes = self._in_role_order(self._min_sizes, order)
        self._crop_mode_checks = self._in_role_order(
            self._crop_mode_checks, order)

    def _refresh_slot_rows(self) -> None:
        """Let an open crop-settings dialog catch up with the slots.

        Both halves of it: rows for controls built since the dialog was laid
        out, and the gate that hides the slots the count does not ask for.
        """
        dialog = getattr(self, "_crop_settings_dialog", None)
        if dialog is not None:
            dialog.refresh_organelle_slots()

    def set_organelle_count(self, count) -> None:
        """How many organelle slots the crop settings should offer.

        The same rule Mask and the Mask live preview follow: the run declares
        `number_of_organelles`, and every panel shows that many. Without it the
        crop settings offered a fixed four -- so a one-organelle run had three
        mask-slice and three minimum-area fields for objects it does not have,
        and each of them propagates into the settings the run reads.

        IT BUILDS THEM AS WELL AS SHOWING THEM. This used to change only what
        was SHOWN, over 702 slots' worth of controls that `_build_controls`
        had already made -- so the answer to "a run with no organelle" was
        2,117 controls hidden behind a gate. The count is now what brings a
        slot's controls into existence, which is why raising it is the only
        route that has to work.
        """
        try:
            wanted = max(0, min(int(count), MAX_ORGANELLES))
        except (TypeError, ValueError):
            return
        if wanted == getattr(self, "_organelle_count", None):
            return
        self._organelle_count = wanted
        self._build_slot_controls(wanted)
        self._refresh_slot_rows()

    def _refresh_control_gates(self, *_args) -> None:
        """Enable each control only when the current crop mode reads it."""
        self._lo_pct.setEnabled(self._normalise.isChecked())
        self._hi_pct.setEnabled(self._normalise.isChecked())
        self._normalize_by.setEnabled(self._normalise.isChecked())
        self._dilate_ratio.setEnabled(self._dilate.isChecked())
        self._buffer.setEnabled(self._use_bbox.isChecked())

    def _png_size_pair(self) -> tuple:
        """The crop size as ``(width, height)``, for the cropper.

        One number on screen, because width and height were always the same
        setting: ``crop_size`` maps onto ``png_size`` in
        :mod:`spacr.picture_settings`, and a scalar ``png_size`` already means
        a square crop in :mod:`spacr.crops`. A settings file that carries a
        non-square pair keeps it, and that pair is what the preview cuts to.

        :returns: ``(width, height)``.
        """
        if self._non_square_png_size:
            return (int(self._non_square_png_size[0]),
                    int(self._non_square_png_size[1]))
        side = int(self._crop_size.value())
        return (side, side)

    def _apply_png_size(self, value) -> None:
        """Take ``png_size`` from a settings file, square or not.

        :param value: a number, or a ``[width, height]`` pair.

        A pair whose sides differ is not squared behind the user's back: it is
        remembered, the width is shown in the box, and the status line says so
        once, because silently changing the shape of somebody's saved crops is
        worse than an explanation.
        """
        self._non_square_png_size = None
        try:
            if isinstance(value, (list, tuple)):
                width, height = int(value[0]), int(value[1])
                if width != height:
                    self._non_square_png_size = (width, height)
                    self._status.setText(
                        f"This settings file crops {width}x{height}. The box "
                        f"shows the width; change it to make crops square.")
            else:
                width = int(value)
        except (TypeError, ValueError, IndexError):
            return
        # Loading a saved rectangle is not a user edit that requests a
        # square. Keep its geometry while setting the displayed width.
        from PySide6.QtCore import QSignalBlocker

        with QSignalBlocker(self._crop_size):
            self._crop_size.setValue(width)
        self._crop_size_timer.start()

    def _on_crop_size_changed(self, _value: int) -> None:
        """Re-crop at the new size, once the typing has stopped.

        The crop size reaches the crops themselves, so it has
        to trigger the same refresh the other crop controls do. It is
        debounced because a spinner passes through 1, 12 and 128 on the way to
        1280, and each of those would otherwise re-cut every object.
        """
        self._non_square_png_size = None
        self._crop_size_timer.start()

    def _on_object_changed(self, name: str) -> None:
        """Re-preview for a different object type."""
        check = self._crop_mode_checks.get(name)
        if check is not None:
            check.setChecked(True)
        self._maybe_propagate()

    def _on_setting_changed(self, *_args) -> None:
        """Re-preview after any control moves."""
        if self._data is not None:
            self.refresh()
        self._maybe_propagate()

    def _maybe_propagate(self, *_args) -> None:
        """Push the tuned settings to the run, if propagation is on."""
        if self._propagate_btn.isChecked():
            self.propagate_settings()

    def _on_propagate_toggled(self, on: bool) -> None:
        """Turn propagation on or off.

        :param on: True to push settings to the run as they change.
        """
        if on:
            self.propagate_settings()


    def _dropped_path(self, event) -> Optional[str]:
        """The usable path out of a drop, or None.

        :param event: the Qt drop event.
        :returns: the path, or None when the drop carries nothing usable.
        """
        mime = event.mimeData()
        if not mime.hasUrls():
            return None
        for url in mime.urls():
            if (
                url.isLocalFile()
                and Path(url.toLocalFile()).suffix.lower() in _SUPPORTED
            ):
                return url.toLocalFile()
        return None

    def dragEnterEvent(self, event):  # noqa: N802
        """Accept a drag carrying something this preview can measure.

        :param event: the Qt drag event.
        """
        event.acceptProposedAction() if self._dropped_path(event) else event.ignore()

    def dragMoveEvent(self, event):  # noqa: N802
        """Keep accepting while droppable input stays over the panel.

        :param event: the Qt drag event.
        """
        event.acceptProposedAction() if self._dropped_path(event) else event.ignore()

    def dropEvent(self, event):  # noqa: N802
        """Take the dropped input and preview it.

        :param event: the Qt drop event.
        """
        path = self._dropped_path(event)
        if path:
            event.acceptProposedAction()
            self.load_array_async(path)
        else:
            event.ignore()

    def _pick_file(self) -> None:
        """Ask for a file to preview."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a merged .npy array", "", "NumPy arrays (*.npy)")
        if path:
            self.load_array_async(path)

    def _load_the_pasted_path(self) -> None:
        """Load whatever was typed or pasted beside the Choose button.

        A file, a `merged/` folder or a run folder all work --
        :func:`resolve_merged_source` decides which, on the worker.
        """
        text = self._paste_box.text().strip()
        if text:
            self.load_array_async(text)

    def _auto_load_from_src(self, src: str) -> bool:
        """Show a field from ``src`` without being asked.

        WHY AUTOMATICALLY. The preview exists to answer "will this crop size
        cut the cell in half" before a run, and it answered nothing until the
        user had found and chosen one of fifty-two arrays by hand. `src` is
        already the folder the run will read, and it is typed in anyway.

        ONLY WHEN NOTHING IS LOADED, and only once per `src`. Re-loading on
        every settings change would throw away an array the user picked
        deliberately -- and would re-randomise the field under them each time
        they touched an unrelated setting.

        :returns: whether a load was started.
        """
        text = str(src or "").strip()
        if not text or text == getattr(self, "_auto_loaded_src", ""):
            return False
        if getattr(self, "_data", None) is not None:
            self._auto_loaded_src = text
            return False
        self._auto_loaded_src = text
        return self.load_array_async(text)

    @property
    def _loads_in_flight(self) -> List[int]:
        """Outstanding loads and re-crops, as a list so ``not ...`` reads well."""
        runner = getattr(self, "_jobs", None)
        return [] if runner is None else [0] * runner.pending_jobs()

    def load_array_async(self, path: str, *,
                         enumerate_sets: bool = True) -> bool:
        """Read ``path`` on a worker, then install it on the GUI thread.

        Every GUI entry point -- the drop handler, the Choose-array dialog and
        the FOV dropdown -- comes through here. A 17 MB merged array is not a
        cheap read, and the crop pass that follows it is far worse.

        :returns: ``True`` when a job was submitted.
        """
        text = str(path).strip() if path else ""
        if not text:
            return False
        self._load_token += 1
        token = self._load_token
        self._status.setText(f"Loading {os.path.basename(text)}…")
        self._jobs.submit(
            lambda: load_merged_array(text, enumerate_sets),
            lambda payload, _t=token: self._on_array_loaded(_t, payload))
        return True

    def _on_array_loaded(self, token: int, payload) -> None:
        """Install a loaded array. Always on the GUI thread."""
        if token != self._load_token or not isinstance(payload, dict):
            return
        sets = payload.get("sets")
        if sets is not None:
            self._sampler.adopt(payload.get("directory"), sets,
                                payload.get("channels") or [])
        if payload.get("error"):
            self._status.setText(payload["error"])
            return
        data = payload.get("data")
        if data is None:
            return
        self._install_array(payload["path"], data)

    def load_array(self, path: str) -> bool:
        """Synchronously read and install one merged array.

        The sibling of ``LivePreviewPanel.load_image``: for programmatic
        callers and tests. The GUI uses :meth:`load_array_async`.
        """
        payload = load_merged_array(path)
        if payload["error"]:
            self._status.setText(payload["error"])
            return False
        self._install_array(path, payload["data"])
        return True

    def _install_array(self, path: str, data: np.ndarray) -> None:
        """Adopt an already-read array and re-crop from it."""
        self._data = data
        self._data_path = path
        self._path_label.setText(
            f"{os.path.basename(path)}  ·  shape {data.shape}")
        for widget in self._mask_dims.values():
            if widget.value() >= data.shape[2]:
                widget.setValue(-1)
        self._refresh_source_selectors()
        self.refresh()

    def shutdown(self) -> None:
        """Abandon anything in flight and leave no QThread behind."""
        runner = getattr(self, "_jobs", None)
        if runner is not None:
            runner.shutdown()

    def closeEvent(self, event):  # noqa: N802
        """Stop any preview work before going away.

        :param event: the Qt close event.
        """
        self.shutdown()
        super().closeEvent(event)


    def _refresh_source_selectors(self) -> None:
        """Re-fill the sets and channel dropdowns for the loaded array.

        The sets dropdown lists a bounded random sample, not the whole folder
        — see :class:`~spacr.qt.widgets.preview_controls.ImageSetSampler`. A
        measure run's ``merged`` folder holds one array per field of view, so
        a 384-well plate puts thousands of entries in here.
        """
        if self._data_path:
            self._sampler.enumerate(Path(self._data_path).parent, _SUPPORTED)
        self._sample_note = apply_sample_to_combo(
            self._fov_box, self._max_sets_box, self._sampler,
            self._data_path, tooltip="Field of view")
        channels = int(self._data.shape[2]) if self._data is not None else 0
        populate_channel_combo(self._channel_box, channels)

    def sample_note(self) -> str:
        """The sentence stating this preview is a sample of N of M sets."""
        return getattr(self, "_sample_note", "")

    def _on_max_sets_changed(self, value: int) -> None:
        """Draw a new sample at the user's new cap — without re-enumerating."""
        if not self._sampler.set_max(int(value)):
            return
        self._refresh_source_selectors()
        if not self._sampler.total:
            return
        self._status.setText(
            self.sample_note()[:1].upper() + self.sample_note()[1:])

    def _on_fov_changed(self, *_args) -> None:
        """Load the field of view the user picked from the dropdown."""
        if self._loading_fov:
            return
        path = self._fov_box.currentData()
        if not path or str(path) == str(self._data_path):
            return
        self._loading_fov = True
        try:
            self.load_array_async(path, enumerate_sets=False)
        finally:
            self._loading_fov = False

    def display_channel(self) -> Optional[int]:
        """Channel index the crops are rendered from, or ``None`` for all."""
        return selected_channel(self._channel_box)

    def _on_display_channel_changed(self, *_args) -> None:
        """Re-render the crop grid from the newly selected channel."""
        self.refresh()


    def _selected_crop_modes(self) -> List[str]:
        """Which crop modes are ticked.

        :returns: the mode names.
        """
        selected = [
            name for name, widget in self._crop_mode_checks.items()
            if widget.isChecked()
        ]
        return selected or [self._object_box.currentText()]

    def settings_for_propagation(self) -> dict:
        """The settings this preview would hand to the real run.

        What makes a preview worth doing: the numbers tuned here are the
        numbers the run uses, rather than something the user must retype.

        :returns: the settings dict.
        """
        normalize: Any = False
        if self._normalise.isChecked():
            normalize = [float(self._lo_pct.value()), float(self._hi_pct.value())]
        return {
            "experiment": self._experiment.text().strip() or "exp",
            "channels": _parse_channels(self._measurement_channels.text()),
            **{f"{name}_mask_dim": _optional_spin_value(widget)
               for name, widget in self._mask_dims.items()},
            "cytoplasm": self._cytoplasm.isChecked(),
            "plot": self._plot.isChecked(),
            "test_mode": self._test_mode.isChecked(),
            "timelapse": self._timelapse.isChecked(),
            "save_png": self._save_png.isChecked(),
            "save_arrays": self._save_arrays.isChecked(),
            "crop_mode": self._selected_crop_modes(),
            "png_size": (list(self._non_square_png_size)
                         if self._non_square_png_size
                         else int(self._crop_size.value())),
            "png_channel_mapping": self._png_channel_mapping(),
            "use_bounding_box": self._use_bbox.isChecked(),
            "normalize": normalize,
            "normalize_by": self._normalize_by.currentText(),
            "dialate_pngs": self._dilate.isChecked(),
            "dialate_png_ratios": [float(self._dilate_ratio.value())],
            **{self._size_floor_key(name): int(widget.value())
               for name, widget in self._min_sizes.items()},
            "uninfected": self._uninfected.isChecked(),
            "merge_edge_pathogen_cells":
                self._merge_edge_pathogen_cells.isChecked(),
        }

    @staticmethod
    def _size_floor_key(name: str) -> str:
        """The settings key holding ``name``'s size floor.

        Organelle's is spelled `_min_area`; every other object still spells
        it `_min_size`. The two names were one setting asked twice, and the
        organelle spelling was retired -- so writing `organelle_min_size`
        here would set a key the run does not read, and this control would
        propagate a value that is silently discarded. That is the same fault
        the `png_dims` comment above records, in the same dictionary.

        :param name: the object, as `_min_sizes` keys it.
        :returns: the settings key to read and write.
        """
        return (f"{name}_min_area" if name.startswith("organelle")
                else f"{name}_min_size")

    def _png_channel_mapping(self) -> Dict[str, Optional[int]]:
        """The RGB control, as the ``{r, g, b}`` mapping the run reads."""
        return dict(self._png_dims.get_value())

    def apply_settings(self, settings: dict) -> None:
        """Seed the panel from the main Measure settings dict.

        The inverse of :meth:`settings_for_propagation`, and tested as one.
        This panel had no ``apply_settings`` at all, so the crop preview --
        opened to decide whether a crop size will cut the cell in half, or
        which stain lands in which colour -- always answered for its own
        defaults rather than for the run about to happen.

        Every field is copied independently: a settings file carrying one
        unusable value must not cost the panel every field after it.
        """
        settings = dict(settings or {})

        # THE COUNT FIRST, because it is what brings the slot controls into
        # existence: a value written into a slot whose control does not exist
        # yet is a value dropped on the floor. Absent is still LEFT ALONE --
        # a dict that mentions no slot and no count is not claiming the run
        # has none, it is making no claim, which is the rule `_set` below
        # follows for every other field.
        speaks_to_the_count = (
            settings.get("number_of_organelles") is not None
            or any(organelle_role_of(key) is not None for key in settings))
        if speaks_to_the_count:
            self.set_organelle_count(organelle_count(settings))
        # AND THE SLOTS THE FILE CARRIES BEYOND IT. `declared_organelle_roles`
        # is the wider of the two -- the slots shown, plus any further slot
        # this dict already has keys for -- so a file written at seven and
        # opened at two keeps controls for slots three to seven and hands
        # their values back untouched instead of dropping them.
        self._build_slot_controls(_slots_the_settings_speak_for(settings))
        self._refresh_slot_rows()

        def _set(fn, key, cast=None):
            """Apply one setting, skipping keys that are absent or None.

            Absent and None are LEFT ALONE rather than applied as a default: the
            preview is showing what the run will do, and filling a gap here would
            show a value the run does not have.
            """
            if key not in settings or settings[key] is None:
                return
            try:
                fn(settings[key] if cast is None else cast(settings[key]))
            except Exception:
                LOG.debug("apply_settings: %r is not usable for %r",
                          settings[key], key, exc_info=True)

        _set(self._experiment.setText, "experiment", str)
        _set(self._measurement_channels.setText, "channels",
             lambda v: ",".join(str(int(c)) for c in v))
        for name in self._object_names():
            if name == "cytoplasm":
                continue
            key = f"{name}_mask_dim"
            if key in settings:
                try:
                    value = settings[key]
                    self._mask_dims[name].setValue(
                        -1 if value is None else int(value))
                except Exception:
                    LOG.debug("apply_settings: bad %s", key, exc_info=True)
            _set(self._min_sizes[name].setValue,
                 self._size_floor_key(name), int)
        _set(self._min_sizes["cytoplasm"].setValue, "cytoplasm_min_size", int)

        for widget, key in (
                (self._cytoplasm, "cytoplasm"),
                (self._plot, "plot"),
                (self._test_mode, "test_mode"),
                (self._timelapse, "timelapse"),
                (self._save_png, "save_png"),
                (self._save_arrays, "save_arrays"),
                (self._use_bbox, "use_bounding_box"),
                (self._dilate, "dialate_pngs"),
                (self._uninfected, "uninfected"),
                (self._merge_edge_pathogen_cells, "merge_edge_pathogen_cells"),
        ):
            _set(widget.setChecked, key, bool)

        if settings.get("crop_mode"):
            modes = {str(m) for m in settings["crop_mode"]}
            for name, widget in self._crop_mode_checks.items():
                widget.setChecked(name in modes)
        _set(self._apply_png_size, "png_size", lambda v: v)
        _set(self._dilate_ratio.setValue, "dialate_png_ratios",
             lambda v: float(list(v)[0]))
        _set(self._normalize_by.setCurrentText, "normalize_by", str)

        if "normalize" in settings:
            value = settings["normalize"]
            if isinstance(value, (list, tuple)) and len(value) == 2:
                self._normalise.setChecked(True)
                try:
                    self._lo_pct.setValue(float(value[0]))
                    self._hi_pct.setValue(float(value[1]))
                except Exception:
                    LOG.debug("apply_settings: bad normalize percentiles",
                              exc_info=True)
            elif value is not None:
                self._normalise.setChecked(bool(value))

        if "png_channel_mapping" in settings or "png_dims" in settings:
            self._png_dims.set_value(_resolve_png_mapping(settings))

        if settings.get("src"):
            self._auto_load_from_src(settings["src"])

    def set_propagate_callback(self, callback) -> None:
        """Set what to call when the user pushes these settings to the run.

        :param callback: called with the settings dict.
        """
        self._propagate_cb = callback

    def propagate_settings(self) -> None:
        """Push the tuned settings to the run, if anything is listening."""
        if self._propagate_cb is None:
            return
        try:
            self._propagate_cb(self.settings_for_propagation())
        except Exception:
            LOG.debug("crop-preview propagation failed", exc_info=True)


    def _current_mask_dim(self) -> Optional[int]:
        """Which mask dimension the selected object uses.

        :returns: the dimension index.
        """
        name = self._object_box.currentText()
        if name == "cytoplasm":
            name = "cell"
        return _optional_spin_value(self._mask_dims[name])

    def _presence(
        self,
        object_name: str,
        cell_region: np.ndarray,
    ) -> Optional[bool]:
        """Widget-reading wrapper over :func:`_presence_in`."""
        if self._data is None:
            return None
        return _presence_in(
            self._data, _optional_spin_value(self._mask_dims[object_name]),
            cell_region, int(self._min_sizes[object_name].value()))

    @staticmethod
    def _phenotype_text(name: str, value: Optional[bool]) -> str:
        """The phenotype label for one crop, for its caption.

        :param name: the phenotype column's name.
        :param value: the object's value in it.
        :returns: the label text.
        """
        return _phenotype_label(name, value)

    def _category_params(self) -> Dict[str, Any]:
        """Snapshot every widget value :func:`annotate_crops` needs.

        Taken on the GUI thread and handed to the worker as plain data. The
        worker must never read a widget.
        """
        #: The three companions the crop grid groups cells by. `organelle`
        #: is the FIRST SLOT and a run may declare none, in which case it has
        #: no controls to read: `annotate_crops` reads these with `.get`, and
        #: a missing dimension renders as "Organelle n/a" -- which is what a
        #: run with no organelle should say.
        companions = [name for name in ("nucleus", "pathogen", "organelle")
                      if name in self._mask_dims]
        return {
            "object": self._object_box.currentText(),
            "cell_dim": _optional_spin_value(self._mask_dims["cell"]),
            "dims": {name: _optional_spin_value(self._mask_dims[name])
                     for name in companions},
            "minima": {name: int(self._min_sizes[name].value())
                       for name in companions},
            "uninfected": bool(self._uninfected.isChecked()),
        }

    def _annotate_cell_categories(self) -> None:
        """Group the crops by phenotype so the grid can head each block."""
        annotate_crops(self._crops, self._data, self._category_params())

    def _preview_blocked_reason(self) -> str:
        """Why this panel cannot crop right now, or ``""``."""
        if self._data is None:
            return self.PREVIEW_SOURCE_HINT
        return ""

    def _extra_work_in_flight(self) -> bool:
        """The crop pass runs on the shared runner, not on a ``_worker``."""
        runner = getattr(self, "_jobs", None)
        return bool(runner is not None and runner.active_jobs())

    def _cancel_extra_work(self) -> None:
        """Drop the result of the crop (or load) pass in flight."""
        self._crop_token += 1
        runner = getattr(self, "_jobs", None)
        if runner is not None:
            runner.cancel()

    def run_preview(self) -> None:
        """Re-crop on demand — the shared name for the shared action.

        Every live view answers to ``run_preview``; this panel's own
        :meth:`refresh` stays as the internal path the crop knobs drive,
        which supersedes rather than refusing.
        """
        reason = self.preview_blocked_reason()
        if reason:
            self.set_preview_status(reason)
            return
        self.refresh()

    def refresh(self) -> None:
        """Re-crop the loaded array and redraw the grid.

        Dispatches: the crop pass runs on a worker and the grid is rebuilt when
        it lands. Every knob in the Crop settings dialog is wired to this, so
        it used to freeze the window for 1441 ms per spinbox step on a
        1024x1024x8 array. Re-cropping supersedes by token, so dragging a
        spinbox through ten values draws the last one rather than all ten.

        Says why when it cannot: returning in silence left the button doing
        nothing with nothing on the status line, which is the one thing no
        live view may do.
        """
        if self._data is None:
            self.set_preview_status(self.PREVIEW_SOURCE_HINT)
            return
        channels = _mapping_to_rgb_list(self._png_channel_mapping())
        channels = [c for c in channels if 0 <= c < self._data.shape[2]]
        one = self.display_channel()
        if one is not None and 0 <= one < self._data.shape[2]:
            channels = [one, one, one]
        if not channels:
            self._status.setText("PNG channels do not exist in this array.")
            return
        mask_dim = self._current_mask_dim()
        if mask_dim is None or mask_dim >= self._data.shape[2]:
            self._status.setText(
                f"No {self._object_box.currentText()} mask slice is configured.")
            self._crops = []
            self._render_grid()
            return

        crop_kwargs = dict(
            mask_dim=mask_dim,
            channels=channels,
            min_area=int(self._min_sizes[self._object_box.currentText()].value()),
            max_area=int(self._max_area.value()),
            mask_background=not self._use_bbox.isChecked(),
            normalize=self._normalise.isChecked(),
            percentiles=(
                float(self._lo_pct.value()), float(self._hi_pct.value())
            ),
            buffer=int(self._buffer.value()),
            limit=int(self._max_crops.value()),
            size=self._png_size_pair(),
        )
        data = self._data
        params = self._category_params()
        self._crop_token += 1
        token = self._crop_token
        self.set_preview_busy(True)
        self._jobs.submit(
            lambda: compute_crops(data, crop_kwargs, params),
            lambda result, _t=token: self._on_crops_ready(_t, result))

    def _on_crops_ready(self, token: int, result) -> None:
        """Draw the crop grid. Always on the GUI thread -- QPixmap demands it."""
        if token != self._crop_token or not isinstance(result, dict):
            return
        self.set_preview_busy(False)
        if result.get("error"):
            self._status.setText(result["error"])
            self.preview_ready.emit(None)
            return
        self._crops = result.get("crops") or []
        self._selected.clear()
        self._render_grid()
        groups = len({entry.get("category") for entry in self._crops})
        self._status.setText(
            f"{len(self._crops)} object(s) · {groups} categor"
            f"{'y' if groups == 1 else 'ies'}")
        self._maybe_propagate()
        self.preview_ready.emit(self._crops)


    def _clear_grid(self) -> None:
        """Empty the thumbnail grid and release its pixmaps.

        RELEASED EXPLICITLY: a preview can hold hundreds of crops, and
        leaving them to the garbage collector keeps a plate's worth of image
        data alive across every re-preview.
        """
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _category_header(self, text: str, entries: List[tuple[int, dict]]) -> QLabel:
        """One heading row for a phenotype block.

        :param text: the heading.
        :returns: the header widget.
        """
        included = sum(bool(entry.get("included", True)) for _, entry in entries)
        excluded = len(entries) - included
        suffix = f"  ·  {included} kept"
        if excluded:
            suffix += f"  ·  {excluded} excluded by filters"
        label = QLabel(text + suffix)
        label.setObjectName("CropCategoryHeader")
        label.setContentsMargins(8, 5, 8, 5)
        try:
            from ..theme import active_palette
            palette = active_palette()
            label.setStyleSheet(
                "QLabel#CropCategoryHeader {"
                f"background: {palette['surface_hi']};"
                f"color: {palette['fg']};"
                f"border-left: 3px solid {palette['accent']};"
                "font-weight: 600; border-radius: 4px;"
                "}"
            )
        except Exception:
            pass
        return label

    def _on_preview_scale(self, scale: float) -> None:
        """Size the crop thumbnails with the preview's own scale.

        :param scale: the preview scale; the thumbnails are 132 px at 100 %.
        """
        self._thumb_px = max(8, int(round(132 * float(scale))))
        if self._crops:
            self._render_grid()

    def _render_grid(self) -> None:
        """Draw the crops, grouped and headed by phenotype."""
        self._clear_grid()
        if not self._crops:
            return
        columns = max(
            1, self._grid_scroll.viewport().width() // (self._thumb_px + 12)
        )
        grouped: Dict[str, List[tuple[int, dict]]] = defaultdict(list)
        if self._object_box.currentText() == "cell" and self._group_cells.isChecked():
            for index, entry in enumerate(self._crops):
                grouped[entry.get("category", "Unclassified")].append(
                    (index, entry))
        else:
            grouped[self._object_box.currentText().capitalize()] = list(
                enumerate(self._crops))

        row = 0
        for category in sorted(grouped):
            entries = grouped[category]
            self._grid.addWidget(
                self._category_header(category, entries),
                row, 0, 1, columns,
            )
            row += 1
            for offset, (index, entry) in enumerate(entries):
                thumb = _CropThumb(
                    index, included=bool(entry.get("included", True)))
                thumb.setPixmap(self._crop_pixmap(entry["crop"]))
                status = "kept" if entry.get("included", True) else "excluded"
                thumb.setToolTip(
                    f"label {entry['label']} · {entry['area']} px² · "
                    f"{entry.get('category', '')} · {status}")
                thumb.clicked.connect(self._on_thumb_clicked)
                self._grid.addWidget(
                    thumb, row + offset // columns, offset % columns)
            row += (len(entries) + columns - 1) // columns

    def _crop_pixmap(self, crop: np.ndarray) -> QPixmap:
        """One crop as a pixmap, scaled for the grid.

        :param crop: the crop's pixels.
        :returns: the pixmap.
        """
        array = np.ascontiguousarray(crop.astype(np.uint8))
        primaries = self.display_primaries()
        if primaries != "rgb" and array.ndim == 3 and array.shape[2] >= 3:
            from ...crops import apply_display_primaries
            array = np.ascontiguousarray(
                apply_display_primaries(array, primaries))
        height, width = array.shape[:2]
        image = QImage(
            array.data, width, height, 3 * width, QImage.Format_RGB888)
        pixmap = scaled_for(QPixmap.fromImage(image.copy()), self,
                            self._thumb_px)
        return _rounded_pixmap(pixmap, radius=8)

    def _on_thumb_clicked(self, index: int) -> None:
        """Open the full-size crop behind a thumbnail.

        :param index: which crop was clicked.
        """
        if not 0 <= index < len(self._crops):
            return
        if index in self._selected:
            self._selected.discard(index)
        else:
            self._selected.add(index)
        entry = self._crops[index]
        selected = (
            f" · {len(self._selected)} selected" if self._selected else "")
        self._status.setText(
            f"label {entry['label']} · {entry['area']} px² · "
            f"{entry.get('category', '')}{selected}")

    def current_params(self) -> dict:
        """The parameters the preview is using right now.

        :returns: the parameters as a plain dict.
        """
        values = self.settings_for_propagation()
        values["n_crops"] = len(self._crops)
        values["selected"] = sorted(self._selected)
        values["categories"] = [
            entry.get("category") for entry in self._crops
        ]
        values["display_channel"] = self.display_channel()
        values["fov"] = self._fov_box.currentText()
        return values


class CropSettingsDialog(QDialog):
    """Tabbed live settings dialog for :class:`MeasurePreviewPanel`.

    :param panel: the preview panel this dialog edits. It is also the
        dialog's PARENT, and the widgets the dialog lays out belong to the
        panel rather than to it -- the dialog only knows which rows they sit
        on, which is what lets a morphology change re-gate them.
    """

    def _insert_organelle_row(self, form: QFormLayout, role: str,
                              label: str, widget: QWidget) -> None:
        """Put one slot row back inside the organelle block.

        AT THE TAIL OF THE BLOCK, not at the foot of the form: the mask
        slices are followed by "Measure cytoplasm" and the size floors by
        "Cytoplasm minimum area", so appending would file Organelle 3 under
        the cytoplasm.

        :param form: the form to insert into.
        :param role: the slot the row belongs to.
        :param label: the row's caption.
        :param widget: the control.
        """
        at = self._organelle_tail.get(id(form), form.rowCount())
        form.insertRow(at, label, widget)
        self._organelle_tail[id(form)] = at + 1
        self._organelle_rows.append((role, form, widget))
        widget.show()

    def _adopt_new_slot_controls(self) -> bool:
        """Lay out the slot controls built since this dialog was.

        `MeasurePreviewPanel.set_organelle_count` is what BUILDS a slot's
        controls, and it can be called while this dialog is on screen -- the
        Measure form's `number_of_organelles` is a live setting. A control
        with no row is as invisible as one that was never made.

        :returns: whether anything was laid out.
        """
        # The last of the three layouts this reaches, so a call made while
        # the dialog is still being built finds nothing half-laid-out.
        if getattr(self, "_filter_form", None) is None:
            return False
        panel = self._panel
        added = False
        for role, widget in panel._mask_dims.items():
            if role not in _ORGANELLE_SLOTS or role in self._mask_rows:
                continue
            self._mask_rows.add(role)
            self._insert_organelle_row(
                self._general_form, role,
                f"{organelle_label(role)} mask slice", widget)
            added = True
        for role, widget in panel._min_sizes.items():
            if role not in _ORGANELLE_SLOTS or role in self._floor_rows:
                continue
            self._floor_rows.add(role)
            self._insert_organelle_row(
                self._filter_form, role,
                f"{role.capitalize()} minimum area", widget)
            added = True
        order = list(panel._crop_mode_checks)
        for role, widget in panel._crop_mode_checks.items():
            if role in self._mode_rows:
                continue
            self._mode_rows.add(role)
            self._mode_layout.insertWidget(order.index(role), widget)
            widget.show()
            added = True
        if added:
            self._install_tooltips()
        return added

    def refresh_organelle_slots(self) -> None:
        """Show one organelle slot per slot the run declares.

        The rows are HIDDEN, not removed: the widgets keep their values, so
        lowering the count and raising it again finds the old answers still
        there -- the same promise `spacr.settings._set_organelle_defaults`
        makes for the settings themselves.

        IT LAYS OUT AS WELL AS GATING. The panel builds a slot's controls
        when the count reaches it rather than building all 702 up front, so
        a count that rises while this dialog is open arrives as controls
        with no rows; `_adopt_new_slot_controls` puts them in the block
        before the gate below decides which are shown.

        An unset count shows every slot that exists, which is what this
        dialog did before the count reached it: better to offer a field too
        many than to hide one a run is using.
        """
        self._adopt_new_slot_controls()
        count = getattr(self._panel, "_organelle_count", None)
        for role, form, widget in getattr(self, "_organelle_rows", ()):
            try:
                wanted = count is None or organelle_number(role) <= count
                position = form.getWidgetPosition(widget)[0]
                if position >= 0:
                    form.setRowVisible(position, bool(wanted))
            except Exception:                                # noqa: BLE001
                LOG.debug("could not gate the %s rows", role, exc_info=True)

    def __init__(self, panel: MeasurePreviewPanel):
        """Build the crop-settings dialog over one preview panel.

        :param panel: the preview these settings belong to.
        """
        super().__init__(panel)
        self._panel = panel
        self.setWindowTitle("Crop preview settings")
        outer = QVBoxLayout(self)
        tabs = QTabWidget(self)
        outer.addWidget(tabs, 1)

        for widget in panel._managed_widgets():
            widget.show()

        general = QWidget()
        form = QFormLayout(general)
        form.addRow("Experiment", panel._experiment)
        form.addRow("Measured channels", panel._measurement_channels)
        form.addRow("Preview object", panel._object_box)
        #: Rows belonging to an organelle slot, so the declared count can
        #: hide the ones a run does not have. Recorded as they are added:
        #: hiding a row needs the FORM as well as the widget.
        self._organelle_rows: List[tuple] = []
        #: Where the next slot row goes in each form, per form: the end of
        #: the organelle block rather than the end of the form.
        self._organelle_tail: Dict[int, int] = {}
        #: Which slots already have a row, per kind, so a slot built later
        #: is laid out once and only once.
        self._mask_rows: set = set()
        self._floor_rows: set = set()
        self._mode_rows: set = set()
        self._general_form = form
        for name, widget in panel._mask_dims.items():
            label = (organelle_label(name) if name in ORGANELLE_ROLES
                     else name.capitalize())
            form.addRow(f"{label} mask slice", widget)
            if name in ORGANELLE_ROLES:
                self._organelle_rows.append((name, form, widget))
                self._mask_rows.add(name)
            if name not in _AFTER_THE_SLOTS:
                self._organelle_tail[id(form)] = form.rowCount()
        form.addRow("Measure cytoplasm", panel._cytoplasm)
        form.addRow("Plot run diagnostics", panel._plot)
        form.addRow("Test mode", panel._test_mode)
        form.addRow("Timelapse", panel._timelapse)
        tabs.addTab(general, "General")

        crops = QWidget()
        crops_form = QFormLayout(crops)
        crops_form.addRow("Save PNG crops", panel._save_png)
        crops_form.addRow("Save raw arrays", panel._save_arrays)
        mode_group = QGroupBox("Crop modes")
        mode_layout = QVBoxLayout(mode_group)
        self._mode_layout = mode_layout
        for name, widget in panel._crop_mode_checks.items():
            mode_layout.addWidget(widget)
            self._mode_rows.add(name)
        crops_form.addRow(mode_group)
        crops_form.addRow("Crop size", panel._crop_size)
        crops_form.addRow("RGB channel order", panel._png_dims)
        crops_form.addRow("Use bounding box", panel._use_bbox)
        crops_form.addRow("Bounding-box padding", panel._buffer)
        crops_form.addRow("Normalise crops", panel._normalise)
        crops_form.addRow("Lower percentile", panel._lo_pct)
        crops_form.addRow("Upper percentile", panel._hi_pct)
        crops_form.addRow("Normalise by", panel._normalize_by)
        crops_form.addRow("Dilate crop masks", panel._dilate)
        crops_form.addRow("Dilation ratio", panel._dilate_ratio)
        tabs.addTab(crops, "Object crops")

        filters = QWidget()
        filter_form = QFormLayout(filters)
        filter_form.addRow("Keep uninfected cells", panel._uninfected)
        filter_form.addRow(
            "Merge edge-pathogen cells", panel._merge_edge_pathogen_cells)
        self._filter_form = filter_form
        for name, widget in panel._min_sizes.items():
            filter_form.addRow(f"{name.capitalize()} minimum area", widget)
            if name in ORGANELLE_ROLES:
                self._organelle_rows.append((name, filter_form, widget))
                self._floor_rows.add(name)
            if name not in _AFTER_THE_SLOTS:
                self._organelle_tail[id(filter_form)] = filter_form.rowCount()
        tabs.addTab(filters, "Filter settings")

        preview = QWidget()
        preview_form = QFormLayout(preview)
        preview_form.addRow("Maximum preview area", panel._max_area)
        preview_form.addRow("Maximum preview crops", panel._max_crops)
        preview_form.addRow("Group cell phenotypes", panel._group_cells)
        tabs.addTab(preview, "Preview")

        self.refresh_organelle_slots()

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        run = QPushButton("Refresh crops")
        run.clicked.connect(panel.refresh)
        buttons.addButton(run, QDialogButtonBox.ActionRole)
        buttons.addButton(panel._propagate_btn, QDialogButtonBox.ActionRole)
        buttons.rejected.connect(self.close)
        outer.addWidget(buttons)
        panel._refresh_control_gates()
        self._install_tooltips()
        self.resize(620, 720)

    def _install_tooltips(self) -> None:
        """Give every row on this dialog its API help.

        SPLIT OUT SO IT CAN BE RE-RUN. A row laid out by
        `_adopt_new_slot_controls` after the dialog was built has had no
        tooltip pass at all, and the pass reads the panel's control dicts,
        so running it again picks the new rows up and leaves the rest as
        they were.
        """
        from ..screens.settings_model import install_api_tooltips

        panel = self._panel
        widget_keys = {
            panel._experiment: "experiment",
            panel._measurement_channels: "channels",
            panel._object_box: "crop_mode",
            **{widget: f"{name}_mask_dim"
               for name, widget in panel._mask_dims.items()},
            panel._cytoplasm: "cytoplasm",
            panel._plot: "plot",
            panel._test_mode: "test_mode",
            panel._timelapse: "timelapse",
            panel._save_png: "save_png",
            panel._save_arrays: "save_arrays",
            panel._crop_size: "png_size",
            panel._png_dims: "png_channel_mapping",
            panel._use_bbox: "use_bounding_box",
            panel._buffer: "bounding_box_padding",
            panel._normalise: "normalize",
            panel._lo_pct: "lower_percentile",
            panel._hi_pct: "upper_percentile",
            panel._normalize_by: "normalize_by",
            panel._dilate: "dialate_pngs",
            panel._dilate_ratio: "dialate_png_ratios",
            panel._uninfected: "uninfected",
            panel._merge_edge_pathogen_cells:
                "merge_edge_pathogen_cells",
            panel._max_area: "preview_max_area",
            panel._max_crops: "preview_max_crops",
            panel._group_cells: "preview_group_cells",
        }
        for name, widget in panel._crop_mode_checks.items():
            widget_keys[widget] = "crop_mode"
        for name, widget in panel._min_sizes.items():
            widget_keys[widget] = panel._size_floor_key(name)
        install_api_tooltips(self, "measure", widget_keys)

    def closeEvent(self, event):
        """Remember the dialog's geometry before it goes.

        :param event: the Qt close event.
        """
        for widget in self._panel._managed_widgets():
            widget.setParent(self._panel)
            widget.hide()
        super().closeEvent(event)
