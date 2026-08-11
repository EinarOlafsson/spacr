"""Settings for the Gate Editor, in tabs.

Split by what the setting is ABOUT, not by which widget reads it:

``General``
    How much data is loaded and how the cloud is drawn. The sampling setting
    lives here because it is the answer to the module being laggy on a real
    dataset -- a screen of a million objects is slow to draw, slow to
    hit-test, and no more informative than a fifth of them.
``2D``
    Gating on a scatter: the tools, the shapes, and the clustering that
    proposes shapes for you.
``3D``
    Gating in a volume. The settings are here; the workspace they drive is
    the next piece of work.

Every setting is a field on :class:`GateEditorSettings`, which is a plain
frozen dataclass -- no Qt -- so what the editor does with a setting can be
tested without building a dialog, and so the whole set can be written to disk
as one thing later.

**Sampling is a display concern only.** A gate drawn on 20% of the objects is
still a statement about all of them; the export re-reads the full table and
applies the gate there (:func:`spacr.filters.gate_mask_over_table`). Nothing
downstream of the plot ever sees the sample.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Mapping, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QLabel, QPushButton, QSpinBox, QTabWidget, QVBoxLayout,
    QWidget,
)

from ..theme import SPACING
from .toggle import Toggle

#: Colour maps offered for the density/colour axis. Perceptually uniform ones
#: first: on a scatter of a million objects the colour IS the reading, and
#: `jet` puts a bright band in the middle of a flat distribution.
COLOUR_MAPS: Tuple[str, ...] = (
    "viridis", "plasma", "magma", "inferno", "cividis", "turbo",
    "Greys", "Blues", "Reds", "coolwarm", "RdBu_r",
)

#: Ways to colour the points. A column name may be used instead.
COLOUR_BY: Tuple[str, ...] = ("density", "flat")

#: Axis scales offered. All are matplotlib SCALES, which change how values are
#: laid out and never what they are -- so a gate drawn before the change still
#: selects the same objects after it.
AXIS_SCALES: Tuple[str, ...] = ("linear", "log", "symlog", "logit")

#: How the cloud is drawn once there are more points than pixels.
RESOLUTION_MODES: Tuple[str, ...] = ("points", "hexbin", "histogram", "density")

#: What a gate is drawn in.
GATE_MODES: Tuple[str, ...] = ("2D", "3D", "xD")

#: Keys a filter can be merged back on. Defaults are what spaCR joins object
#: tables and png_list on; the object label alone is not unique across fields.
MERGE_KEYS: Tuple[str, ...] = (
    "plateID", "rowID", "columnID", "fieldID", "object_label",
)

#: Clustering algorithms the picker offers, RE-EXPORTED from the module that
#: implements them so this dialog cannot list one the code does not have.
#:
#: That is not hypothetical tidiness. This tuple used to be written here and
#: read "dbscan", "hdbscan", "kmeans", while `cluster_gates` called DBSCAN
#: whatever it said -- so choosing k-means returned DBSCAN's answer under
#: another name. k-means is not in the list any more rather than newly
#: written, because it has to be told the number of clusters and the gate
#: editor has no such setting; adding one to justify a list entry is the
#: wrong way round, and the tooltip beside this control already argues
#: against k-means on a cytometry scatter.
from .gate_spec import CLUSTER_METHODS  # noqa: E402  (re-export)


@dataclass(frozen=True)
class GateEditorSettings:
    """Everything the Gate Editor is configured by.

    Frozen: settings are replaced wholesale rather than mutated, so a screen
    can compare what it has against what it is given and re-read the table
    only when something that actually costs a read has changed.
    """

    # -- general ----------------------------------------------------------
    #: Fraction of the table loaded, in (0, 1]. The lag fix: gates are drawn
    #: on this, and applied to everything on export.
    sample_fraction: float = 1.0
    #: Hard row cap after sampling. 0 means none. Ten thousand by default:
    #: past that a scatter is drawing more markers than the screen has pixels,
    #: and the large-data raster kicks in and takes the per-point settings
    #: with it.
    max_points: int = 10_000
    colour_map: str = "viridis"
    #: What the colour map is applied TO. "density" is the default because a
    #: cytometry scatter has no colour axis and the overlap is the reading.
    #: Any column name is also valid.
    colour_by: str = "density"
    #: Axis scales. These are DISPLAY transforms -- matplotlib scales -- so a
    #: gate's coordinates keep meaning the measurement they were drawn on. A
    #: transform that rewrote the values would silently invalidate every gate
    #: already drawn, which is why z-score and min-max are not offered here.
    x_scale: str = "linear"
    y_scale: str = "linear"
    point_size: float = 6.0
    point_opacity: float = 0.6
    #: How the cloud is rendered. "points" is one marker per object; the rest
    #: bin first, which is what makes a million objects draw at all.
    resolution_mode: str = "points"
    #: Bins per axis when binning. Not a "resolution" in pixels: the bin is
    #: the unit the data is summarised into, and saying so in data terms is
    #: what makes the same setting mean the same thing at any zoom.
    bins: int = 200
    show_grid: bool = False
    #: Kept so a saved settings set from before x_scale/y_scale still asks
    #: for a log axis. `x_scale` wins when both are set: someone who chose a
    #: scale has said what they mean.
    log_x: bool = False
    log_y: bool = False

    def scale_for(self, axis: str) -> str:
        """The scale for "x" or "y", honouring the retired log flags."""
        chosen = getattr(self, f"{axis}_scale", "linear")
        if chosen != "linear":
            return chosen
        return "log" if getattr(self, f"log_{axis}", False) else "linear"

    # -- 2D ---------------------------------------------------------------
    default_tool: str = "rectangle"
    gate_line_width: float = 0.5
    #: Ring the gated objects, rather than only outlining the gate.
    highlight_gated: bool = True
    #: Keys a filter is merged back onto the object tables with.
    merge_keys: Tuple[str, ...] = MERGE_KEYS
    #: Magic wand: how far apart two objects can be and still be
    #: neighbours, in scaled units. Small enough to stop at a gap.
    wand_tolerance: float = 0.05
    #: How far from the click the wand may reach at all. Without a ceiling a
    #: single chain of objects bridging two populations merges them.
    wand_max_radius: float = 0.35
    cluster_method: str = "dbscan"
    cluster_eps: float = 0.5
    cluster_min_samples: int = 20
    cluster_scale: bool = True
    #: Search the clustering hyperparameters instead of taking the values
    #: above. What "Walk" means everywhere else in spaCR: try the space, show
    #: each result as it arrives, let the user pick.
    cluster_walk: bool = False
    cluster_walk_steps: int = 12

    # -- 3D ---------------------------------------------------------------
    gate_mode: str = "2D"
    #: How xD projects. PCA is always available; the others need a package.
    reduction: str = "pca"
    #: How many components xD produces. Three, so the 3D view has a Z.
    components: int = 3
    #: What a merge does with a primary object that has no children.
    merge_na: str = "keep"
    #: Which object everything else is rolled up onto. Decides what a row of
    #: the merged table MEANS, so it is a choice rather than an assumption.
    merge_primary: str = "cell"
    #: column -> aggregation, beating the rules worked out from the column's
    #: name. A default that is right most of the time is a wrong answer nobody
    #: can find the rest of the time, so every one of them is overridable.
    merge_overrides: Mapping[str, str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "merge_overrides",
                           dict(self.merge_overrides or {}))
    z_axis: str = ""
    #: Voxels per axis in the 3D workspace.
    voxel_bins: int = 64
    #: Snap the camera to the nearest axis when a spin ends, so a 3D gate is
    #: always finally read from a square-on view.
    snap_to_axis: bool = True
    spin_speed: float = 1.0

    def replaced(self, **changes) -> "GateEditorSettings":
        """A copy with ``changes`` applied."""
        return replace(self, **changes)

    def costs_a_reload(self, other: "GateEditorSettings") -> bool:
        """Whether moving to ``other`` means re-reading the table.

        Only two settings do. Everything else is drawing, and re-reading a
        large table because the user changed a colour map is exactly the kind
        of lag this dialog exists to remove.
        """
        return (self.sample_fraction != other.sample_fraction
                or self.max_points != other.max_points)


class GateSettingsDialog(QDialog):
    """The tabbed settings window.

    Emits :attr:`settings_changed` as the user edits, so the settings that
    CAN be applied live are, without waiting for OK. The ones that cannot --
    sampling, which means re-reading the table -- are applied by the screen
    when it sees them change, which is why the signal carries the whole
    settings object rather than one field.
    """

    #: The settings changed. Carries a whole :class:`GateEditorSettings`.
    settings_changed = Signal(object)
    #: The per-column aggregation table was asked for. The dialog does not own
    #: it: only the screen knows which columns are loaded.
    aggregation_rules_requested = Signal()

    def __init__(self, settings: GateEditorSettings, parent=None, *,
                 columns: Tuple[str, ...] = ()):
        super().__init__(parent)
        self.setWindowTitle("Gate editor settings")
        self.setObjectName("GateSettingsDialog")
        self._settings = settings
        self._live = True

        outer = QVBoxLayout(self)
        outer.setSpacing(SPACING["sm"])
        self.tabs = QTabWidget(self)
        self.tabs.setObjectName("GateSettingsTabs")
        outer.addWidget(self.tabs, 1)

        self.tabs.addTab(self._general_tab(), "General")
        self.tabs.addTab(self._two_d_tab(), "2D")
        self.tabs.addTab(self._three_d_tab(columns), "3D")

        buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        buttons.rejected.connect(self.accept)
        buttons.accepted.connect(self.accept)
        outer.addWidget(buttons)

    # -- the tabs ---------------------------------------------------------
    def _general_tab(self) -> QWidget:
        page = QWidget(self)
        form = QFormLayout(page)

        self._sample = QSpinBox(page)
        self._sample.setRange(1, 100)
        self._sample.setSuffix(" %")
        self._sample.setValue(int(round(self._settings.sample_fraction * 100)))
        self._sample.setToolTip(
            "How much of the table to load. Drawing and hit-testing a million "
            "objects is what makes this module slow; a fifth of them is the "
            "same cloud. Gates are applied to EVERY object on export, whatever "
            "fraction was loaded.")
        self._sample.valueChanged.connect(
            lambda v: self._change(sample_fraction=max(0.01, v / 100.0)))
        form.addRow("Load", self._sample)

        self._max_points = QSpinBox(page)
        self._max_points.setRange(0, 10_000_000)
        self._max_points.setSingleStep(10_000)
        self._max_points.setSpecialValueText("no cap")
        self._max_points.setValue(self._settings.max_points)
        self._max_points.setToolTip(
            "A hard ceiling applied after the percentage, for the case where "
            "even a small fraction of a very large table is too much.")
        self._max_points.valueChanged.connect(
            lambda v: self._change(max_points=int(v)))
        form.addRow("At most", self._max_points)

        self._cmap = QComboBox(page)
        self._cmap.addItems(COLOUR_MAPS)
        self._cmap.setCurrentText(self._settings.colour_map)
        self._cmap.setToolTip(
            "On a dense scatter the colour is the reading, so the "
            "perceptually uniform maps come first — a map with a bright band "
            "in it invents a feature in a flat distribution.")
        self._cmap.currentTextChanged.connect(
            lambda v: self._change(colour_map=v))
        form.addRow("Colour map", self._cmap)

        self._resolution = QComboBox(page)
        self._resolution.addItems(RESOLUTION_MODES)
        self._resolution.setCurrentText(self._settings.resolution_mode)
        self._resolution.setToolTip(
            "How the cloud is drawn once there are more objects than pixels. "
            "Binning first (hexbin, histogram, density) is what lets a very "
            "large table draw at all, and shows where the objects actually "
            "are — overplotted points hide their own density.")
        self._resolution.currentTextChanged.connect(
            lambda v: self._change(resolution_mode=v))
        form.addRow("Data resolution", self._resolution)

        self._bins = QSpinBox(page)
        self._bins.setRange(10, 2000)
        self._bins.setValue(self._settings.bins)
        self._bins.setToolTip(
            "Bins per axis when binning. In data terms, not pixels, so the "
            "same setting means the same thing at any zoom.")
        self._bins.valueChanged.connect(lambda v: self._change(bins=int(v)))
        form.addRow("Bins", self._bins)

        self._point_size = QDoubleSpinBox(page)
        self._point_size.setRange(0.5, 60.0)
        self._point_size.setSingleStep(0.5)
        self._point_size.setValue(self._settings.point_size)
        self._point_size.valueChanged.connect(
            lambda v: self._change(point_size=float(v)))
        form.addRow("Point size", self._point_size)

        self._opacity = QDoubleSpinBox(page)
        self._opacity.setRange(0.02, 1.0)
        self._opacity.setSingleStep(0.05)
        self._opacity.setValue(self._settings.point_opacity)
        self._opacity.setToolTip(
            "Below 1 the overlap itself shows density, which is the cheapest "
            "way to read a crowded scatter.")
        self._opacity.valueChanged.connect(
            lambda v: self._change(point_opacity=float(v)))
        form.addRow("Point opacity", self._opacity)

        self._colour_by = QComboBox(page)
        self._colour_by.setEditable(True)
        self._colour_by.addItems(COLOUR_BY)
        self._colour_by.setCurrentText(self._settings.colour_by)
        self._colour_by.setToolTip(
            "What the colour map is applied to. 'density' colours each point "
            "by how crowded it is, which is the reading a scatter of a "
            "million objects actually carries; 'flat' is one colour. Any "
            "column name also works.")
        self._colour_by.currentTextChanged.connect(
            lambda v: self._change(colour_by=v))
        form.addRow("Colour by", self._colour_by)

        for label, field, current in (("X scale", "x_scale",
                                       self._settings.x_scale),
                                      ("Y scale", "y_scale",
                                       self._settings.y_scale)):
            box = QComboBox(page)
            box.addItems(AXIS_SCALES)
            box.setCurrentText(current)
            box.setToolTip(
                "How the axis is laid out. These change the spacing, never "
                "the values, so a gate drawn before the change still selects "
                "the same objects. 'log' is skipped on a measurement that "
                "reaches zero, where it would draw nothing at all.")
            box.currentTextChanged.connect(
                lambda v, f=field: self._change(**{f: v}))
            setattr(self, f"_{field}", box)
            form.addRow(label, box)

        self._grid = Toggle("Show grid", page)
        self._grid.setChecked(self._settings.show_grid)
        self._grid.toggled.connect(lambda v: self._change(show_grid=bool(v)))
        form.addRow("", self._grid)

        return page

    def _two_d_tab(self) -> QWidget:
        page = QWidget(self)
        form = QFormLayout(page)

        self._tool = QComboBox(page)
        self._tool.addItems(("rectangle", "ellipse", "polygon", "threshold",
                            "wand"))
        self._tool.setCurrentText(self._settings.default_tool)
        self._tool.setToolTip(
            "The tool a drag uses when nothing else is armed. Rectangle, "
            "because dragging a box is what everyone tries first.")
        self._tool.currentTextChanged.connect(
            lambda v: self._change(default_tool=v))
        form.addRow("Default tool", self._tool)

        self._line_width = QDoubleSpinBox(page)
        self._line_width.setRange(0.1, 6.0)
        self._line_width.setSingleStep(0.2)
        self._line_width.setValue(self._settings.gate_line_width)
        self._line_width.valueChanged.connect(
            lambda v: self._change(gate_line_width=float(v)))
        form.addRow("Gate line width", self._line_width)

        self._highlight = Toggle("Ring the gated objects", page)
        self._highlight.setChecked(self._settings.highlight_gated)
        self._highlight.setToolTip(
            "Marks the objects inside each shown gate. The rest of the cloud "
            "stays on screen either way — a gate highlights, it never hides.")
        self._highlight.toggled.connect(
            lambda v: self._change(highlight_gated=bool(v)))
        form.addRow("", self._highlight)

        self._merge_boxes: Dict[str, Toggle] = {}
        merge_note = QLabel(
            "Keys a filter is merged back onto the object tables with. The "
            "object label alone repeats in every field, so dropping a key "
            "merges objects that are not the same object.", page)
        merge_note.setWordWrap(True)
        form.addRow("Merge on", merge_note)
        for key in MERGE_KEYS:
            box = Toggle(key, page)
            box.setChecked(key in self._settings.merge_keys)
            box.toggled.connect(self._on_merge_key_toggled)
            self._merge_boxes[key] = box
            form.addRow("", box)

        self._wand_tolerance = QDoubleSpinBox(page)
        self._wand_tolerance.setRange(0.001, 1.0)
        self._wand_tolerance.setDecimals(3)
        self._wand_tolerance.setSingleStep(0.01)
        self._wand_tolerance.setValue(self._settings.wand_tolerance)
        self._wand_tolerance.setToolTip(
            "Magic wand: how far apart two objects can be and still count as "
            "neighbours. This is what makes it a watershed rather than a "
            "circle — the selection flows along a dense ridge and stops at a "
            "gap, so a bent population comes out whole.")
        self._wand_tolerance.valueChanged.connect(
            lambda v: self._change(wand_tolerance=float(v)))
        form.addRow("Wand tolerance", self._wand_tolerance)

        self._wand_radius = QDoubleSpinBox(page)
        self._wand_radius.setRange(0.01, 5.0)
        self._wand_radius.setSingleStep(0.05)
        self._wand_radius.setValue(self._settings.wand_max_radius)
        self._wand_radius.setToolTip(
            "How far from the click the wand may reach at all. Without a "
            "ceiling, one chain of objects bridging two populations merges "
            "them — which on a real scatter happens more often than not.")
        self._wand_radius.valueChanged.connect(
            lambda v: self._change(wand_max_radius=float(v)))
        form.addRow("Wand max distance", self._wand_radius)

        self._cluster_method = QComboBox(page)
        self._cluster_method.addItems(CLUSTER_METHODS)
        self._cluster_method.setCurrentText(self._settings.cluster_method)
        self._cluster_method.setToolTip(
            "DBSCAN finds clusters of any shape and is not told how many "
            "there are — which is the whole problem with k-means on a "
            "cytometry scatter.")
        self._cluster_method.currentTextChanged.connect(
            lambda v: self._change(cluster_method=v))
        form.addRow("Clustering", self._cluster_method)

        self._eps = QDoubleSpinBox(page)
        self._eps.setRange(0.01, 100.0)
        self._eps.setSingleStep(0.05)
        self._eps.setValue(self._settings.cluster_eps)
        self._eps.valueChanged.connect(
            lambda v: self._change(cluster_eps=float(v)))
        form.addRow("eps", self._eps)

        self._min_samples = QSpinBox(page)
        self._min_samples.setRange(2, 10_000)
        self._min_samples.setValue(self._settings.cluster_min_samples)
        self._min_samples.valueChanged.connect(
            lambda v: self._change(cluster_min_samples=int(v)))
        form.addRow("min samples", self._min_samples)

        self._scale = Toggle("Standardise the axes first", page)
        self._scale.setChecked(self._settings.cluster_scale)
        self._scale.setToolTip(
            "Without it, eps means a distance in whichever measurement has "
            "the larger numbers and the other axis is ignored.")
        self._scale.toggled.connect(
            lambda v: self._change(cluster_scale=bool(v)))
        form.addRow("", self._scale)

        self._walk = Toggle("Walk", page)
        self._walk.setChecked(self._settings.cluster_walk)
        self._walk.setToolTip(
            "Try the space instead of taking the values above, showing each "
            "result as it arrives so you can pick the one that matches what "
            "you can see.")
        self._walk.toggled.connect(lambda v: self._change(cluster_walk=bool(v)))
        form.addRow("", self._walk)

        self._walk_steps = QSpinBox(page)
        self._walk_steps.setRange(2, 200)
        self._walk_steps.setValue(self._settings.cluster_walk_steps)
        self._walk_steps.valueChanged.connect(
            lambda v: self._change(cluster_walk_steps=int(v)))
        form.addRow("Walk steps", self._walk_steps)
        return page

    def _three_d_tab(self, columns: Tuple[str, ...]) -> QWidget:
        page = QWidget(self)
        form = QFormLayout(page)

        self._reduction = QComboBox(page)
        self._reduction.addItems(("pca", "umap", "tsne"))
        self._reduction.setCurrentText(self._settings.reduction)
        self._reduction.setToolTip(
            "How xD projects many measurements onto few. PCA is always "
            "available and is the only one whose axes have a stated meaning "
            "— the share of variance each component explains.")
        self._reduction.currentTextChanged.connect(
            lambda v: self._change(reduction=v))
        form.addRow("xD projection", self._reduction)

        self._components = QSpinBox(page)
        self._components.setRange(2, 10)
        self._components.setValue(self._settings.components)
        self._components.valueChanged.connect(
            lambda v: self._change(components=int(v)))
        form.addRow("Components", self._components)

        self._merge_primary = QComboBox(page)
        self._merge_primary.addItems(
            ("cell", "nucleus", "pathogen", "cytoplasm", "organelle"))
        self._merge_primary.setCurrentText(self._settings.merge_primary)
        self._merge_primary.setToolTip(
            "The object everything else is rolled up onto. It decides what a "
            "row of the merged table means — rolling cells onto pathogens is "
            "a legitimate thing to want and gives a different table.")
        self._merge_primary.currentTextChanged.connect(
            lambda v: self._change(merge_primary=v))
        form.addRow("Merge: primary object", self._merge_primary)

        self._merge_na = QComboBox(page)
        self._merge_na.addItems(("keep", "zero", "drop"))
        self._merge_na.setCurrentText(self._settings.merge_na)
        self._merge_na.setToolTip(
            "What happens to an object with no children when tables are "
            "merged. A cell with no pathogens genuinely has a pathogen COUNT "
            "of zero, and genuinely has no pathogen mean intensity at all — "
            "so 'keep' leaves that blank rather than inventing a zero.")
        self._merge_na.currentTextChanged.connect(
            lambda v: self._change(merge_na=v))
        form.addRow("Merge: missing children", self._merge_na)

        self._mode = QComboBox(page)
        self._mode.addItems(GATE_MODES)
        self._mode.setCurrentText(self._settings.gate_mode)
        self._mode.setToolTip(
            "2D gates on a scatter, 3D in a volume, xD on more measurements "
            "than can be drawn at once.")
        self._mode.currentTextChanged.connect(lambda v: self._change(gate_mode=v))
        form.addRow("Gate in", self._mode)

        self._z = QComboBox(page)
        self._z.setEditable(True)
        self._z.addItem("")
        self._z.addItems(columns)
        self._z.setCurrentText(self._settings.z_axis)
        self._z.currentTextChanged.connect(lambda v: self._change(z_axis=v))
        form.addRow("Z", self._z)

        self._voxels = QSpinBox(page)
        self._voxels.setRange(8, 512)
        self._voxels.setValue(self._settings.voxel_bins)
        self._voxels.setToolTip(
            "Voxels per axis. A volume is bins cubed, so this costs far more "
            "than the same number does in 2D.")
        self._voxels.valueChanged.connect(
            lambda v: self._change(voxel_bins=int(v)))
        form.addRow("Voxels", self._voxels)

        self._snap = Toggle("Snap to the nearest axis when a spin ends", page)
        self._snap.setChecked(self._settings.snap_to_axis)
        self._snap.setToolTip(
            "So a 3D gate is always finally read square-on. A volume stopped "
            "at an arbitrary angle cannot be read off at all.")
        self._snap.toggled.connect(lambda v: self._change(snap_to_axis=bool(v)))
        form.addRow("", self._snap)

        self._spin = QDoubleSpinBox(page)
        self._spin.setRange(0.1, 10.0)
        self._spin.setSingleStep(0.1)
        self._spin.setValue(self._settings.spin_speed)
        self._spin.valueChanged.connect(
            lambda v: self._change(spin_speed=float(v)))
        form.addRow("Spin speed", self._spin)

        self._rules_button = QPushButton("Aggregation rules…", page)
        self._rules_button.setToolTip(
            "Show the rule chosen for every measurement, and change any of "
            "them. The rules follow what a column MEASURES — areas and counts "
            "sum, a minimum takes the minimum — and a silent default that is "
            "right 95% of the time is a wrong answer nobody can find the "
            "other 5%.")
        self._rules_button.clicked.connect(self.aggregation_rules_requested.emit)
        form.addRow("", self._rules_button)

        note = QLabel(
            "The 3D workspace itself is the next piece of work. These "
            "settings are read by it when it lands; nothing here changes the "
            "2D view.", page)
        note.setWordWrap(True)
        form.addRow("", note)
        return page

    # -- edits ------------------------------------------------------------
    def _on_merge_key_toggled(self, _checked: bool) -> None:
        """Merge keys are one setting, so they are collected, not appended.

        Kept in MERGE_KEYS order rather than click order: the tuple is used as
        a join key list, and a join on the same keys in a different order is
        the same join written two ways.
        """
        chosen = tuple(k for k in MERGE_KEYS if self._merge_boxes[k].isChecked())
        self._change(merge_keys=chosen)

    def _change(self, **fields) -> None:
        self._settings = self._settings.replaced(**fields)
        if self._live:
            self.settings_changed.emit(self._settings)

    def set_mode(self, mode: str) -> None:
        """Show a mode chosen elsewhere, without re-emitting it.

        The 2D/3D/xD buttons and this dropdown are two views of one setting.
        Echoing the change back would be a loop; showing it is what keeps the
        window honest about the state the editor is actually in.
        """
        if mode not in GATE_MODES:
            return
        self._live = False
        try:
            self._mode.setCurrentText(mode)
            self._settings = self._settings.replaced(gate_mode=mode)
        finally:
            self._live = True

    def settings(self) -> GateEditorSettings:
        return self._settings
