"""Restyle a figure that has already been drawn, without re-running anything.

The Figures panel used to offer three controls -- background, text colour,
text size -- so changing a legend, an axis scale or a series colour meant
editing settings and re-running the analysis, or opening the PDF in
Illustrator.

Everything here works on the **live matplotlib Figure**, which is why it can
offer what a saved page cannot. A PDF does allow a stroke to be recoloured or
a font resized, but not anything data-bound: a log axis has to recompute every
position. Working on the Figure gives both, and
:meth:`spacr.qt.widgets.figure_queue.FigureQueue.figure_for` restores an
evicted figure from its spill so an old figure is editable too.

The controls are BUILT FROM THE FIGURE, not from a fixed list. A figure with
no legend gets no legend row; one with three line series gets three colour
pickers. That is what makes "as many settings as possible, depending on the
graph" true rather than aspirational -- and it means a figure type spaCR
grows later is covered without editing this file.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Optional

LOG = logging.getLogger(__name__)

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

#: Axis scales offered. ``symlog`` is included because screen scores are often
#: signed and a plain log drops every non-positive point silently.
AXIS_SCALES = ("linear", "log", "symlog")

LEGEND_LOCATIONS = (
    "best", "upper right", "upper left", "lower left", "lower right",
    "right", "center left", "center right", "lower center", "upper center",
    "center",
)

LINE_STYLES = (("-", "Solid"), ("--", "Dashed"), ("-.", "Dash-dot"),
               (":", "Dotted"), ("None", "None"))


def _as_hex(colour, fallback: str = "#1f77b4") -> str:
    """Any matplotlib colour spec as ``#rrggbb``.

    matplotlib hands back whatever it stored: an RGBA tuple from
    ``patch.get_facecolor()``, a named colour, a float grey, or an ARRAY of
    RGBA rows from a collection. ``QColor`` accepts none of those, and passing
    a tuple raised ``TypeError: QVariant must be holding a QColor`` the moment
    a colour button was clicked -- so every colour control in this dialog was
    dead on arrival.
    """
    try:
        from matplotlib.colors import to_hex
        import numpy as np

        value = colour
        # A collection stores one row per element; they share a colour here.
        if isinstance(value, np.ndarray):
            value = value[0] if value.ndim > 1 and len(value) else value
        elif isinstance(value, (list, tuple)) and len(value) \
                and isinstance(value[0], (list, tuple, np.ndarray)):
            value = value[0]
        return to_hex(value, keep_alpha=False)
    except Exception:  # pragma: no cover - genuinely unreadable colour
        return fallback


def _colour_button(initial, on_pick: Callable[[str], None]) -> QPushButton:
    """A button showing a colour that opens a picker."""
    button = QPushButton()
    state = {"colour": _as_hex(initial)}

    def _paint():
        colour = QColor(state["colour"])
        button.setText(state["colour"])
        if colour.isValid():
            button.setStyleSheet(
                f"background-color: {colour.name()}; "
                f"color: {'#000' if colour.lightness() > 127 else '#fff'};")

    def _choose():
        colour = QColorDialog.getColor(QColor(state["colour"]), button)
        if colour.isValid():
            state["colour"] = colour.name()
            _paint()
            on_pick(colour.name())

    button.clicked.connect(_choose)
    _paint()
    return button


def _series_of(axis):
    """Every restylable series on ``axis``, as ``(label, artist)`` pairs.

    Lines and collections (a scatter is a collection) are what a user means by
    "the data". Named series come first so a legend label is what they are
    picked by rather than an index.
    """
    series = []
    for index, line in enumerate(axis.lines):
        label = line.get_label()
        if not label or label.startswith("_"):
            label = f"line {index + 1}"
        series.append((label, line))
    for index, collection in enumerate(axis.collections):
        label = collection.get_label()
        if not label or label.startswith("_"):
            label = f"points {index + 1}"
        series.append((label, collection))
    return series


class FigureSettingsDialog(QDialog):
    """Every appearance control the given figure can support."""

    #: Milliseconds of quiet before a restyle is drawn. Short, because the
    #: draw now happens on a worker thread and costs the GUI thread ~10 ms --
    #: inside a frame -- so there is nothing left to hide behind a long delay.
    #: It was 220 ms when the render blocked, and that still felt like lag.
    REDRAW_DELAY_MS = 60

    def __init__(self, figure, parent=None, *, on_change: Optional[Callable] = None,
                 propagate_callback: Optional[Callable] = None):
        super().__init__(parent)
        self.setWindowTitle("Figure settings")
        self._figure = figure
        self._on_change = on_change
        self._propagate_cb = propagate_callback
        self.resize(520, 640)

        # A SNAPSHOT TO GO BACK TO. The dialog this replaced restored the
        # figure on Cancel, and said why: "live apply with no way out is a
        # trap: the user drags a spin box to see what it does and there is no
        # longer an 'as it was'". This dialog changes far more than that one
        # did, so the trap is correspondingly worse. The copy is the same one
        # the preview renderer takes, ~14 ms, and buys a working Cancel.
        self._snapshot = None
        try:
            import pickle
            self._snapshot = pickle.dumps(figure)
        except Exception:  # pragma: no cover - artists that will not pickle
            pass

        # Coalesce redraws. Every control calls _changed(); this restarts a
        # single-shot timer, so a burst of twenty value changes costs one
        # render instead of twenty.
        #: True while a render holds the GUI thread; see :meth:`_redraw_now`.
        self._rendering = False
        #: A change arrived mid-render and still needs to reach the picture.
        self._dirty = False
        self._redraw = QTimer(self)
        self._redraw.setSingleShot(True)
        self._redraw.timeout.connect(self._redraw_now)

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget(self)
        layout.addWidget(self.tabs)

        self.tabs.addTab(self._scroll(self._figure_tab()), "Figure")
        for index, axis in enumerate(figure.axes):
            name = axis.get_title() or f"Axes {index + 1}"
            self.tabs.addTab(self._scroll(self._axes_tab(axis)), name[:18])

        # The Image UMAP half (instruction 75): every UMAP setting, live
        # against this figure. Only for a figure carrying the embedding it was
        # drawn from -- without it "live" would mean re-running the reduction
        # and every point would move.
        self._umap_settings = None
        self._umap_payload = getattr(figure, "_spacr_umap_payload", None)
        self._umap_applied = {}
        if isinstance(self._umap_payload, dict):
            self._build_umap_tab()

        # Scrolling the panel must scroll it, not edit whatever is under the
        # pointer. Qt gives spin boxes and combos the wheel by default, so a
        # scroll gesture over this dialog changed a dozen settings and
        # triggered a render for each -- which is what made it unusable.
        self._block_wheel_on_inputs()

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self._propagate_btn = QPushButton("Propagate settings")
        if callable(propagate_callback):
            self._propagate_btn.setToolTip(
                "Write these values into the module's settings panel, so the "
                "next run starts from them and they are saved with it.")
        else:
            self._propagate_btn.setEnabled(False)
            self._propagate_btn.setToolTip(
                "Only available for a figure opened from a module that has a "
                "settings panel to write into.")
        self._propagate_btn.clicked.connect(self._propagate)
        buttons.addButton(self._propagate_btn, QDialogButtonBox.ActionRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------- UMAP, propagate

    def _build_umap_tab(self) -> None:
        """Add every Image UMAP setting, live against this figure."""
        try:
            from .umap_figure_settings import UmapFigureSettings
        except Exception:  # pragma: no cover - UMAP support absent
            return
        values = dict(self._umap_payload.get("settings") or {})
        self._umap_settings = UmapFigureSettings(values, self)
        self._umap_settings.settings_changed.connect(self._on_umap_changed)
        self.tabs.addTab(self._scroll(self._umap_settings), "Image UMAP")
        self._umap_applied = dict(self._umap_settings.values())

    def _on_umap_changed(self, values: dict) -> None:
        """Push a changed Image UMAP setting at the figure, now.

        The embedding is read, never recomputed -- see
        :func:`spacr.qt.widgets.umap_figure_settings.redraw_umap_figure`.
        """
        from .umap_figure_settings import apply_to_figure

        mode = apply_to_figure(self._figure, self._umap_payload, values,
                               self._umap_applied)
        self._umap_applied = dict(values)
        if mode:
            self._changed()

    def umap_values(self) -> dict:
        """Every Image UMAP setting the window holds, or ``{}``."""
        if self._umap_settings is None:
            return {}
        return self._umap_settings.values()

    def _propagate(self) -> None:
        """Send the current values into the module's settings panel."""
        if not callable(self._propagate_cb):
            return
        values = dict(self.umap_values())
        try:
            self._propagate_cb(values)
        except Exception:
            pass

    def reject(self):
        """Put the figure back the way the window found it, then close.

        Live apply with no way out is a trap: the user drags a spin box to see
        what it does and there is no longer an "as it was".
        """
        if self._snapshot is not None:
            try:
                import pickle

                restored = pickle.loads(self._snapshot)
                # Copy the restored state back INTO the figure the queue
                # holds, rather than swapping the object -- everything else
                # refers to the original by identity.
                self._figure.clear()
                for axis in restored.axes:
                    self._figure._axstack.add(axis)
                    axis.figure = self._figure
                    axis.set_figure(self._figure)
                self._figure.patch.set_facecolor(restored.patch.get_facecolor())
                self._figure.set_size_inches(*restored.get_size_inches())
                self._changed()
            except Exception:  # pragma: no cover - restore is best-effort
                pass
        super().reject()

    #: Input types that steal the wheel from the scroll area beneath them.
    _WHEEL_STEALERS = (QSpinBox, QDoubleSpinBox, QComboBox)

    #: Above this many series on one axes, offer colouring RULES instead of a
    #: control per series. A volcano scatters once per compartment -- 27 of
    #: them -- and a control block each is 135 controls that read as styling
    #: individual data points.
    SERIES_DETAIL_LIMIT = 8

    #: Palettes offered as the colouring rule. Qualitative first: a series set
    #: is categorical, and a sequential map implies an order that is not there.
    PALETTES = ("tab10", "tab20", "Set1", "Set2", "Set3", "Dark2", "Paired",
                "Accent", "viridis", "plasma", "cividis", "coolwarm")

    def _add_series_rules(self, form, axis, series) -> None:
        """Colouring rules for an axes with too many series to list.

        The user asked for "coloring rules, but not options for each
        individual datapoint". This is that: one palette across the whole
        series set, and single size/opacity controls that reach all of them.
        """
        form.addRow(QLabel(f"— {len(series)} series —"))
        note = QLabel(
            "Too many series to style one by one, so these rules apply "
            "across all of them.")
        note.setWordWrap(True)
        form.addRow(note)

        palette = QComboBox()
        palette.addItem("Keep current colours", None)
        for name in self.PALETTES:
            palette.addItem(name, name)

        def apply_palette(*_):
            name = palette.currentData()
            if not name:
                return
            import matplotlib as mpl

            colormap = mpl.colormaps[name]
            count = max(len(series), 1)
            for index, (_label, artist) in enumerate(series):
                # A qualitative map is indexed by position; a continuous one
                # is sampled across its range. Using the wrong one gives every
                # series nearly the same colour.
                colour = (colormap(index % colormap.N) if colormap.N <= 32
                          else colormap(index / max(count - 1, 1)))
                try:
                    artist.set_color(colour)
                except Exception:  # pragma: no cover - artist without colour
                    pass
            self._changed()
        palette.currentIndexChanged.connect(apply_palette)
        form.addRow("Palette", palette)

        size = QDoubleSpinBox()
        size.setRange(1.0, 600.0)
        size.setValue(36.0)

        def apply_size(value):
            for _label, artist in series:
                if hasattr(artist, "set_sizes"):
                    artist.set_sizes([value])
                elif hasattr(artist, "set_markersize"):
                    artist.set_markersize(value ** 0.5)
            self._changed()
        size.valueChanged.connect(apply_size)
        form.addRow("Point size (all)", size)

        opacity = QDoubleSpinBox()
        opacity.setRange(0.05, 1.0)
        opacity.setSingleStep(0.05)
        opacity.setValue(1.0)

        def apply_opacity(value):
            for _label, artist in series:
                artist.set_alpha(value)
            self._changed()
        opacity.valueChanged.connect(apply_opacity)
        form.addRow("Opacity (all)", opacity)

        edge = QDoubleSpinBox()
        edge.setRange(0.0, 5.0)
        edge.setSingleStep(0.1)
        edge.setValue(0.0)

        def apply_edge(value):
            for _label, artist in series:
                if hasattr(artist, "set_linewidth"):
                    artist.set_linewidth(value)
            self._changed()
        edge.valueChanged.connect(apply_edge)
        form.addRow("Outline width (all)", edge)

    def _block_wheel_on_inputs(self) -> None:
        """Let inputs take the wheel only once they are deliberately focused.

        ``findChildren`` takes ONE type per call in PySide6, not a tuple, so
        this loops -- passing a tuple raises TypeError and the whole dialog
        fails to construct.
        """
        for kind in self._WHEEL_STEALERS:
            for widget in self.findChildren(kind):
                widget.setFocusPolicy(Qt.StrongFocus)
                widget.installEventFilter(self)

    def eventFilter(self, obj, event):  # noqa: N802 - Qt name
        if (event.type() == QEvent.Wheel
                and isinstance(obj, self._WHEEL_STEALERS)
                and not obj.hasFocus()):
            event.ignore()
            return True
        return super().eventFilter(obj, event)

    def closeEvent(self, event):  # noqa: N802 - Qt name
        """Land any pending redraw at full quality before going away."""
        if self._redraw.isActive():
            self._redraw.stop()
            self._redraw_now(preview=False)
        else:
            self._redraw_now(preview=False)
        super().closeEvent(event)

    # ------------------------------------------------------------- plumbing

    @staticmethod
    def _scroll(widget: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(widget)
        return area

    def _changed(self) -> None:
        """Ask for a redraw. Every control calls this.

        Live feedback rather than an OK button, because restyling is a
        judgement made by looking -- 'is this legend small enough yet' is not
        answerable from a dialog that only applies on close. But *immediate*
        feedback is what froze the app: a render rewrites the raster and the
        vector page, and a spin box emits a value per step. So the redraw is
        debounced, and the one that lands mid-edit is a cheap preview.
        """
        self._redraw.start(self.REDRAW_DELAY_MS)

    def _redraw_now(self, preview: bool = True) -> None:
        if self._on_change is None:
            return
        # RENDERS MUST NOT STACK.
        #
        # A preview blocks the GUI thread for ~150 ms. Qt keeps delivering
        # events during that render -- spin-box auto-repeat, wheel, the timer
        # itself -- and without this guard each one lands another render behind
        # the current one. The queue grows faster than it drains and the window
        # stops responding: the hang.
        #
        # Instead a request that arrives mid-render only sets a flag, and one
        # final redraw runs afterwards. Interaction stays smooth because the
        # thread is always free between renders, and the picture still ends up
        # matching the controls.
        if self._rendering:
            self._dirty = True
            return
        self._rendering = True
        try:
            try:
                self._on_change(preview=preview)
            except TypeError:
                # A caller that does not know about preview rendering.
                self._on_change()
        finally:
            self._rendering = False
        if self._dirty:
            self._dirty = False
            self._redraw.start(self.REDRAW_DELAY_MS)

    # ----------------------------------------------------------------- tabs

    def _figure_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        figure = self._figure

        def set_face(colour):
            figure.patch.set_facecolor(colour)
            self._changed()
        form.addRow("Background", _colour_button(
            figure.patch.get_facecolor(), set_face))

        width = QDoubleSpinBox()
        width.setRange(1, 60)
        width.setDecimals(1)
        width.setValue(figure.get_figwidth())
        height = QDoubleSpinBox()
        height.setRange(1, 60)
        height.setDecimals(1)
        height.setValue(figure.get_figheight())

        def resize(*_):
            figure.set_size_inches(width.value(), height.value())
            self._changed()
        width.valueChanged.connect(resize)
        height.valueChanged.connect(resize)
        form.addRow("Width (in)", width)
        form.addRow("Height (in)", height)

        dpi = QSpinBox()
        dpi.setRange(50, 1200)
        dpi.setValue(int(figure.get_dpi()))
        dpi.valueChanged.connect(
            lambda value: (figure.set_dpi(value), self._changed()))
        form.addRow("DPI", dpi)

        # One control that reaches every text object at once, because
        # "make the fonts bigger" is a single intention.
        all_text = QSpinBox()
        all_text.setRange(4, 48)
        all_text.setValue(10)

        def set_all_text(size):
            for axis in figure.axes:
                items = [axis.title, axis.xaxis.label, axis.yaxis.label]
                items += axis.get_xticklabels() + axis.get_yticklabels()
                legend = axis.get_legend()
                if legend is not None:
                    items += list(legend.get_texts())
                for item in items:
                    item.set_fontsize(size)
            self._changed()
        all_text.valueChanged.connect(set_all_text)
        form.addRow("All text size", all_text)

        # AND THE COLOUR OF ALL OF IT. There was a size control here and no
        # colour control at all, so the background could be changed and the
        # writing on top of it could not -- which on a dark background is a
        # figure with invisible axes and no way to fix it.
        #
        # Every text object, the same intention as the size row: title, axis
        # labels, BOTH sets of tick labels, the tick marks and spines that
        # frame them, and any legend. Recolouring the labels and leaving the
        # spines is the half-done version that looks like a bug.
        def set_all_text_colour(colour):
            for axis in figure.axes:
                items = [axis.title, axis.xaxis.label, axis.yaxis.label]
                items += axis.get_xticklabels() + axis.get_yticklabels()
                legend = axis.get_legend()
                if legend is not None:
                    items += list(legend.get_texts())
                    title = legend.get_title()
                    if title is not None:
                        items.append(title)
                for item in items:
                    item.set_color(colour)
                # matplotlib spells it `color`; this file spells it `colour`
                # everywhere else. Keep both straight rather than passing an
                # unrecognised keyword, which raises rather than being
                # ignored.
                axis.tick_params(color=colour, labelcolor=colour, which="both")
                for spine in axis.spines.values():
                    spine.set_edgecolor(colour)
            if figure._suptitle is not None:
                figure._suptitle.set_color(colour)
            self._changed()

        current_text_colour = "#000000"
        if figure.axes:
            try:
                current_text_colour = _as_hex(
                    figure.axes[0].xaxis.label.get_color())
            except Exception:      # pragma: no cover - odd colour spec
                pass
        form.addRow("All text colour",
                    _colour_button(current_text_colour, set_all_text_colour))

        suptitle = QLineEdit(
            figure._suptitle.get_text() if figure._suptitle else "")
        suptitle.editingFinished.connect(
            lambda: (figure.suptitle(suptitle.text()), self._changed()))
        form.addRow("Figure title", suptitle)
        return page

    def _axes_tab(self, axis) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)

        title = QLineEdit(axis.get_title())
        title.editingFinished.connect(
            lambda: (axis.set_title(title.text()), self._changed()))
        form.addRow("Title", title)

        for label, getter, setter in (
            ("X label", axis.get_xlabel, axis.set_xlabel),
            ("Y label", axis.get_ylabel, axis.set_ylabel),
        ):
            edit = QLineEdit(getter())
            edit.editingFinished.connect(
                lambda e=edit, s=setter: (s(e.text()), self._changed()))
            form.addRow(label, edit)

        # Scales -- the data-bound controls a saved page could never offer.
        for label, getter, setter in (
            ("X scale", axis.get_xscale, axis.set_xscale),
            ("Y scale", axis.get_yscale, axis.set_yscale),
        ):
            combo = QComboBox()
            combo.addItems(AXIS_SCALES)
            current = getter()
            if current in AXIS_SCALES:
                combo.setCurrentText(current)
            combo.currentTextChanged.connect(
                lambda value, s=setter: (s(value), self._changed()))
            form.addRow(label, combo)

        # Limits. Four boxes and an autoscale switch, because "zoom the
        # volcano to the part with the hits in it" is the single most common
        # thing anyone wants from a plot and there was no way to ask for it.
        for label, getter, setter in (
            ("X limits", axis.get_xlim, axis.set_xlim),
            ("Y limits", axis.get_ylim, axis.set_ylim),
        ):
            low, high = (float(v) for v in getter())
            span = abs(high - low) or 1.0
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            boxes = []
            for value in (low, high):
                box = QDoubleSpinBox()
                # Room to move well outside the data, and enough precision for
                # a log axis where the interesting range can be tiny.
                box.setRange(-1e12, 1e12)
                box.setDecimals(4)
                box.setSingleStep(span / 20.0)
                box.setValue(value)
                box.setKeyboardTracking(False)  # not one redraw per keystroke
                boxes.append(box)
                row_layout.addWidget(box)

            def apply_limits(*_, s=setter, b=boxes):
                lower, upper = b[0].value(), b[1].value()
                if lower == upper:
                    return  # a zero-width axis throws; wait for the other box
                s(lower, upper)
                self._changed()
            for box in boxes:
                box.valueChanged.connect(apply_limits)
            form.addRow(label, row)

        auto = QPushButton("Autoscale to data")

        def do_autoscale():
            axis.relim()
            axis.autoscale()
            self._changed()
        auto.clicked.connect(do_autoscale)
        form.addRow("", auto)

        for label, getter, setter in (
            ("Invert X", axis.xaxis_inverted, axis.invert_xaxis),
            ("Invert Y", axis.yaxis_inverted, axis.invert_yaxis),
        ):
            check = QCheckBox()
            check.setChecked(bool(getter()))
            check.toggled.connect(
                lambda _v, s=setter: (s(), self._changed()))
            form.addRow(label, check)

        # Grid
        grid = QCheckBox()
        grid.setChecked(any(line.get_visible()
                            for line in axis.get_xgridlines()))
        grid_axis = QComboBox()
        grid_axis.addItems(("both", "x", "y"))
        grid_width = QDoubleSpinBox()
        grid_width.setRange(0.1, 6.0)
        grid_width.setSingleStep(0.1)
        grid_width.setValue(0.8)
        grid_colour = {"value": "#cccccc"}

        def apply_grid(*_):
            # Line properties are passed ONLY when enabling. matplotlib warns
            # "First parameter to grid() is false, but line properties are
            # supplied" and then turns the grid ON regardless -- so the
            # unconditional version made the checkbox unable to switch the
            # grid off, which is the opposite of what it says.
            if grid.isChecked():
                axis.grid(True, axis=grid_axis.currentText(),
                          color=grid_colour["value"],
                          linewidth=grid_width.value())
            else:
                axis.grid(False, axis=grid_axis.currentText())
            self._changed()
        grid.toggled.connect(apply_grid)
        grid_axis.currentTextChanged.connect(apply_grid)
        grid_width.valueChanged.connect(apply_grid)
        form.addRow("Grid", grid)
        form.addRow("Grid axis", grid_axis)
        form.addRow("Grid width", grid_width)
        form.addRow("Grid colour", _colour_button(
            grid_colour["value"],
            lambda c: (grid_colour.__setitem__("value", c), apply_grid())))

        # Spines and ticks
        spine_width = QDoubleSpinBox()
        spine_width.setRange(0.0, 10.0)
        spine_width.setSingleStep(0.25)
        spine_width.setValue(
            next(iter(axis.spines.values())).get_linewidth()
            if axis.spines else 1.0)

        def set_spines(value):
            for spine in axis.spines.values():
                spine.set_linewidth(value)
            self._changed()
        spine_width.valueChanged.connect(set_spines)
        form.addRow("Spine width", spine_width)

        hide_top_right = QCheckBox()
        hide_top_right.setChecked(
            not axis.spines["top"].get_visible() if "top" in axis.spines
            else False)

        def set_top_right(hidden):
            for name in ("top", "right"):
                if name in axis.spines:
                    axis.spines[name].set_visible(not hidden)
            self._changed()
        hide_top_right.toggled.connect(set_top_right)
        form.addRow("Hide top/right", hide_top_right)

        tick_size = QSpinBox()
        tick_size.setRange(4, 40)
        labels = axis.get_xticklabels()
        tick_size.setValue(int(labels[0].get_fontsize()) if labels else 10)
        tick_size.valueChanged.connect(
            lambda value: (axis.tick_params(labelsize=value), self._changed()))
        form.addRow("Tick label size", tick_size)

        # Legend -- only offered when there is one, or something to make one
        # from. A legend row on a figure with no labelled series is a control
        # that does nothing.
        handles, _labels = axis.get_legend_handles_labels()
        if axis.get_legend() is not None or handles:
            legend_on = QCheckBox()
            legend_on.setChecked(axis.get_legend() is not None
                                 and axis.get_legend().get_visible())
            legend_where = QComboBox()
            legend_where.addItems(LEGEND_LOCATIONS)
            legend_size = QSpinBox()
            legend_size.setRange(4, 32)
            legend_size.setValue(9)
            legend_cols = QSpinBox()
            legend_cols.setRange(1, 6)
            legend_frame = QCheckBox()
            legend_frame.setChecked(True)

            def apply_legend(*_):
                existing = axis.get_legend()
                if not legend_on.isChecked():
                    if existing is not None:
                        existing.set_visible(False)
                    self._changed()
                    return
                # Rebuilding needs labelled artists. Calling legend() without
                # them warns "No artists with labels found to put in legend"
                # and returns nothing, losing the legend the figure already
                # had -- so an existing legend is restyled in place instead.
                handles, _labels = axis.get_legend_handles_labels()
                if handles:
                    axis.legend(loc=legend_where.currentText(),
                                ncol=legend_cols.value(),
                                frameon=legend_frame.isChecked(),
                                prop={"size": legend_size.value()})
                elif existing is not None:
                    existing.set_visible(True)
                    existing.set_frame_on(legend_frame.isChecked())
                    for text in existing.get_texts():
                        text.set_fontsize(legend_size.value())
                self._changed()
            for control in (legend_on, legend_frame):
                control.toggled.connect(apply_legend)
            legend_where.currentTextChanged.connect(apply_legend)
            legend_size.valueChanged.connect(apply_legend)
            legend_cols.valueChanged.connect(apply_legend)
            form.addRow("Legend", legend_on)
            form.addRow("Legend position", legend_where)
            form.addRow("Legend text size", legend_size)
            form.addRow("Legend columns", legend_cols)
            form.addRow("Legend frame", legend_frame)

        series = _series_of(axis)
        # MANY SERIES GET A RULE, NOT A CONTROL EACH.
        #
        # A volcano scatters once per compartment, so an axis can hold 27
        # collections. One block each is 135 controls and reads as styling
        # individual data points, which is not a thing anyone wants to do to a
        # screen. Past the threshold the dialog offers what actually governs
        # the appearance: a palette applied across the series, and one set of
        # size/opacity controls that reach all of them.
        if len(series) > self.SERIES_DETAIL_LIMIT:
            self._add_series_rules(form, axis, series)
            return page

        # Few enough to be worth naming individually.
        for label, artist in series:
            form.addRow(QLabel(f"— {label} —"))

            def set_colour(colour, a=artist):
                try:
                    a.set_color(colour)
                except Exception:  # pragma: no cover - artist without colour
                    pass
                self._changed()
            try:
                current = artist.get_color()
            except Exception:  # pragma: no cover
                current = "#1f77b4"
            form.addRow("  Colour", _colour_button(current, set_colour))

            if hasattr(artist, "set_linewidth"):
                line_width = QDoubleSpinBox()
                line_width.setRange(0.0, 12.0)
                line_width.setSingleStep(0.25)
                # A collection returns an ARRAY of widths, one per element,
                # not a scalar. float() on it happens to work today and is
                # deprecated; take the first explicitly.
                try:
                    raw = artist.get_linewidth()
                    if hasattr(raw, "__len__") and not isinstance(raw, str):
                        raw = raw[0] if len(raw) else 1.0
                    width_value = float(raw)
                except Exception:  # pragma: no cover
                    width_value = 1.0
                line_width.setValue(width_value)
                line_width.valueChanged.connect(
                    lambda value, a=artist: (a.set_linewidth(value),
                                             self._changed()))
                form.addRow("  Line width", line_width)

            if hasattr(artist, "set_linestyle"):
                style = QComboBox()
                for code, name in LINE_STYLES:
                    style.addItem(name, code)
                style.currentIndexChanged.connect(
                    lambda _i, a=artist, c=style: (
                        a.set_linestyle(c.currentData()), self._changed()))
                form.addRow("  Line style", style)

            if hasattr(artist, "set_markersize"):
                marker = QDoubleSpinBox()
                marker.setRange(0.0, 40.0)
                try:
                    marker.setValue(float(artist.get_markersize()))
                except Exception:  # pragma: no cover
                    marker.setValue(6.0)
                marker.valueChanged.connect(
                    lambda value, a=artist: (a.set_markersize(value),
                                             self._changed()))
                form.addRow("  Marker size", marker)
            elif hasattr(artist, "set_sizes"):
                point = QDoubleSpinBox()
                point.setRange(1.0, 600.0)
                point.setValue(36.0)
                point.valueChanged.connect(
                    lambda value, a=artist: (a.set_sizes([value]),
                                             self._changed()))
                form.addRow("  Point size", point)

            alpha = QDoubleSpinBox()
            alpha.setRange(0.05, 1.0)
            alpha.setSingleStep(0.05)
            try:
                alpha.setValue(float(artist.get_alpha() or 1.0))
            except Exception:  # pragma: no cover
                alpha.setValue(1.0)
            alpha.valueChanged.connect(
                lambda value, a=artist: (a.set_alpha(value), self._changed()))
            form.addRow("  Opacity", alpha)

        return page


def build_figure_context_menu(parent, figure, *, on_change=None,
                              open_settings=None) -> QMenu:
    """The right-click menu for a drawn figure.

    The frequent toggles are one click; everything else is behind
    "Figure settings…". A figure that cannot be restyled -- evicted, and its
    spill unreadable -- gets a menu saying so rather than a menu that silently
    does nothing.
    """
    menu = QMenu(parent)
    if figure is None:
        action = QAction("This figure can no longer be restyled", parent)
        action.setEnabled(False)
        menu.addAction(action)
        return menu

    axes = list(figure.axes)

    def _notify() -> None:
        """Redraw after a menu toggle, CHEAPLY.

        A context-menu toggle is the same kind of edit the settings dialog
        makes, and the dialog learned long ago to preview: a full-quality
        render rewrites the raster AND the vector page, measured at ~263 ms
        on an 823-point volcano, and the user is mid-gesture. Preview here
        too, and let the next full render -- a resize, an export, closing the
        settings dialog -- catch up.

        `on_change` may be a callable that predates preview rendering, so the
        keyword is offered and withdrawn rather than assumed.
        """
        if not on_change:
            return
        try:
            on_change(preview=True)
        except TypeError:
            on_change()

    def _apply(func):
        for axis in axes:
            func(axis)
        _notify()

    legend_present = any(a.get_legend() is not None for a in axes)
    legend_action = QAction("Legend", parent)
    legend_action.setCheckable(True)
    legend_action.setChecked(
        legend_present and all(a.get_legend().get_visible()
                               for a in axes if a.get_legend() is not None))

    def toggle_legend(checked):
        for axis in axes:
            existing = axis.get_legend()
            if existing is not None:
                existing.set_visible(checked)
            elif checked and axis.get_legend_handles_labels()[0]:
                axis.legend()
        _notify()
    legend_action.toggled.connect(toggle_legend)
    menu.addAction(legend_action)

    grid_action = QAction("Grid", parent)
    grid_action.setCheckable(True)
    grid_action.setChecked(any(line.get_visible()
                               for axis in axes
                               for line in axis.get_xgridlines()))
    grid_action.toggled.connect(
        lambda checked: _apply(lambda a: a.grid(checked)))
    menu.addAction(grid_action)

    scales = menu.addMenu("Axis scale")
    for name, setter in (("X", "set_xscale"), ("Y", "set_yscale")):
        submenu = scales.addMenu(name)
        for scale in AXIS_SCALES:
            action = QAction(scale, parent)
            action.triggered.connect(
                lambda _checked=False, s=scale, m=setter:
                _apply(lambda a: getattr(a, m)(s)))
            submenu.addAction(action)

    menu.addSeparator()
    save = QAction("Save figure as…", parent)
    save.setToolTip("Write this figure to a file, with the styling it has "
                    "on screen right now.")
    save.triggered.connect(lambda: save_figure_as(parent, figure))
    menu.addAction(save)

    settings = QAction("Figure settings…", parent)
    if open_settings is not None:
        settings.triggered.connect(lambda: open_settings())
    menu.addAction(settings)
    return menu


def save_figure_as(parent, figure, path: str = "") -> str:
    """Write ``figure`` to a file the user picks. Returns the path, or "".

    Instruction 108 asks for a figure to be savable from the same right-click
    that restyles it, and 119 for "each figure should be editable and
    savable". The saved file is what is ON SCREEN: a restyle that survives to
    the display and not to the file is a restyle the user cannot use, which
    is the whole reason for editing a figure in the first place.

    :param path: bypass the dialog. For tests, and for callers that already
        know where the file goes.
    """
    if figure is None:
        return ""
    if not path:
        from PySide6.QtWidgets import QFileDialog

        path, _filter = QFileDialog.getSaveFileName(
            parent, "Save figure", "figure.png",
            "PNG image (*.png);;PDF document (*.pdf);;"
            "SVG image (*.svg);;All files (*)")
        if not path:
            return ""

    try:
        from ..preferences import (figure_bg_is_transparent, get_figure_colors,
                                   get_figure_png_dpi)
        background, _foreground = get_figure_colors()
        dpi = get_figure_png_dpi()
    except Exception:                    # pragma: no cover - no settings store
        background, dpi = "none", 200

        def figure_bg_is_transparent(value):
            return str(value).lower() in ("none", "transparent")

    try:
        # Vector formats have no meaningful DPI, and passing one makes
        # matplotlib rasterise text in some backends.
        vector = os.path.splitext(path)[1].lower() in (".pdf", ".svg", ".eps")
        figure.savefig(
            path, bbox_inches="tight", facecolor=background,
            transparent=figure_bg_is_transparent(background),
            **({} if vector else {"dpi": dpi}))
    except Exception as error:           # noqa: BLE001 - report, do not raise
        LOG.info("could not save figure to %s: %s", path, error)
        return ""
    return path


__all__ = ["FigureSettingsDialog", "build_figure_context_menu", "AXIS_SCALES",
           "save_figure_as"]
