"""Preview and save a separately styled copy of a Matplotlib figure.

Export styling is applied to a detached copy, leaving the figure displayed in
spaCR unchanged. Figures that cannot be copied still use the ordinary save
path without a styled preview.
"""
import logging
from typing import Optional

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QPushButton,
                               QSpinBox, QVBoxLayout, QWidget)

LOG = logging.getLogger("spacr.qt.save_figure")

#: Why the page size is not editable here for a pyqtgraph plot.
_SIZE_REASON = ("set on the plot's right-click menu, under Canvas: it is one "
                "page size, read by every export this plot makes")

#: Why there is no resolution to set for a pyqtgraph plot.
_RESOLUTION_REASON = ("this plot writes true vector for PDF and SVG, which "
                      "have no resolution, and sizes its PNG from the canvas "
                      "shape on its own right-click menu")


def _reason_label(text: str, reason: str = ""):
    """Return a form label that displays why its control is disabled.

    The reason appears directly in the label and as a tooltip, so it remains
    available when a disabled widget does not receive hover events.
    """
    from PySide6.QtWidgets import QLabel

    if not reason:
        return QLabel(text)
    label = QLabel(f"{text} — {reason}")
    label.setEnabled(False)
    label.setToolTip(reason)
    label.setWordWrap(True)
    return label


#: What the file can be written as.
FORMATS = (("png", "PNG image"), ("pdf", "PDF document"),
           ("svg", "SVG image"), ("tiff", "TIFF image"))

#: What a pyqtgraph plot can be written as. TIFF is absent because
#: `FastPlot.export` routes anything that is not .pdf or .svg through
#: pyqtgraph's ImageExporter, which writes PNG -- offering TIFF would produce
#: a file whose name and contents disagreed.
FAST_PLOT_FORMATS = (("png", "PNG image"), ("pdf", "PDF document"),
                     ("svg", "SVG image"))

#: Ink choices for the file. "As on screen" is first because it is what the
#: previous behaviour did, and a dialog that silently changed the default
#: would restyle every save a user made out of habit.
INKS = (("", "as on screen"),
        ("#231F20", "black ink on white — for paper"),
        ("#FFFFFF", "white ink on transparent — for a dark slide"))


def is_fast_plot(figure) -> bool:
    """Return whether a plot supports spaCR's fast styled-export protocol.

    Parameters
    ----------
    figure : object
        Candidate Matplotlib or pyqtgraph figure.

    Returns
    -------
    bool
        ``True`` when the object provides callable ``styled_snapshot`` and
        ``export_styled`` methods.

    Notes
    -----
    Capability detection allows new fast-plot classes to support this dialog
    without requiring a class registry.
    """
    return (figure is not None
            and callable(getattr(figure, "styled_snapshot", None))
            and callable(getattr(figure, "export_styled", None)))


def copy_figure(figure):
    """Create a detached figure for export preview.

    Parameters
    ----------
    figure : object
        Figure to copy through Python's pickle protocol.

    Returns
    -------
    object or None
        Independent copy of ``figure``. ``None`` is returned when the input is
        absent or contains state that cannot be serialized, such as a live
        canvas or closure.

    Notes
    -----
    The serialization round trip matches the one used by the figure queue.
    Returning ``None`` lets callers fall back to an ordinary save without
    altering the on-screen figure.
    """
    import io
    import pickle

    if figure is None:
        return None
    try:
        buffer = io.BytesIO()
        pickle.dump(figure, buffer)
        buffer.seek(0)
        return pickle.load(buffer)
    except Exception:                                        # noqa: BLE001
        LOG.debug("figure could not be copied for preview", exc_info=True)
        return None


def style_for_file(figure, *, ink: str = "", background: str = "",
                   grid: bool = False, width: float = 0.0,
                   height: float = 0.0, dpi: int = 0,
                   font_scale: float = 0.0):
    """Apply export-only styling to a Matplotlib figure.

    Parameters
    ----------
    figure : matplotlib.figure.Figure or None
        Figure copy to modify. ``None`` is accepted for preview fallbacks.
    ink : str, optional
        Color applied to titles, labels, ticks, spines, legends, and text.
        An empty string preserves the existing colors.
    background : str, optional
        Figure and axes background color. An empty string makes both
        backgrounds transparent.
    grid : bool, default False
        Draw major grid lines when ``True`` and disable them when ``False``.
    width, height : float, default 0
        Output dimensions in inches. The size changes only when both values
        are positive.
    dpi : int, default 0
        Output resolution. Zero preserves the figure's current resolution.
    font_scale : float, default 0
        Multiplier applied to every text artist. Values at or below zero
        preserve the current text sizes.

    Returns
    -------
    matplotlib.figure.Figure or None
        The same figure object after styling, or ``None`` when ``figure`` was
        ``None``.

    Notes
    -----
    This function is intended for detached export copies. It does not read or
    update live figure preferences.
    """
    if figure is None:
        return None
    if width and height:
        figure.set_size_inches(float(width), float(height))
    if dpi:
        figure.set_dpi(int(dpi))
    # Apply the scale to existing artists because rcParams only affect text
    # created after the parameter change.
    if font_scale and font_scale > 0:
        for text in figure.findobj(match=lambda o: hasattr(o, "get_fontsize")):
            try:
                text.set_fontsize(text.get_fontsize() * float(font_scale))
            except Exception:                                # noqa: BLE001
                continue
    if background:
        figure.patch.set_facecolor(background)
        figure.patch.set_alpha(1.0)
    else:
        figure.patch.set_alpha(0.0)
    for axes in figure.axes:
        if background:
            axes.set_facecolor(background)
        else:
            axes.patch.set_alpha(0.0)
        if ink:
            axes.title.set_color(ink)
            axes.xaxis.label.set_color(ink)
            axes.yaxis.label.set_color(ink)
            axes.tick_params(color=ink, labelcolor=ink, which="both")
            for spine in axes.spines.values():
                spine.set_edgecolor(ink)
            legend = axes.get_legend()
            if legend is not None:
                for text in legend.get_texts():
                    text.set_color(ink)
            for text in axes.texts:
                text.set_color(ink)
        # Matplotlib enables the grid when line properties accompany
        # ``grid(False)``. Supply styling arguments only for the enabled case.
        if grid:
            axes.grid(True, which="major", linewidth=0.4, alpha=0.35)
        else:
            axes.grid(False)
    return figure


class SaveFigureDialog(QDialog):
    """Preview and save an independently styled figure copy.

    Parameters
    ----------
    figure : matplotlib.figure.Figure or fast plot
        Source figure. Matplotlib figures are copied before preview styling;
        fast plots provide their own styled snapshot and export methods.
    parent : QWidget, optional
        Parent widget for the modal dialog.

    Notes
    -----
    Export settings never modify the source displayed in the application.
    """

    def __init__(self, figure, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Save figure")
        self._source = figure
        self._preview = None
        self._canvas = None
        #: A pyqtgraph plot is styled and written through its own two methods
        #: rather than through matplotlib's. Decided once, here, so every
        #: branch below reads a flag instead of re-sniffing the object.
        self._fast = is_fast_plot(figure)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.ink = QComboBox()
        for value, label in INKS:
            self.ink.addItem(label, value)
        self.ink.currentIndexChanged.connect(self.refresh)
        form.addRow("ink", self.ink)

        self.background = QComboBox()
        for value, label in (("", "transparent"), ("#FFFFFF", "white"),
                             ("#000000", "black")):
            self.background.addItem(label, value)
        self.background.currentIndexChanged.connect(self.refresh)
        form.addRow("background", self.background)

        self.grid = QCheckBox("draw a grid")
        self.grid.toggled.connect(self.refresh)
        form.addRow("", self.grid)

        # WHAT THE RIGHT-CLICK MENU OFFERS, AND MORE. A file is read at a
        # size and on a page the screen never had, so the things worth
        # changing for it are not the things worth changing on screen:
        # the proportions, how heavy the ink is at print size, and what the
        # axes are CALLED in a figure that has left the application.
        #
        # Each is empty or zero by default, and an untouched control
        # changes nothing -- so a user who wants the figure as it looks
        # still gets it in one click.
        self.aspect = QDoubleSpinBox()
        self.aspect.setRange(0.0, 20.0)
        self.aspect.setDecimals(3)
        self.aspect.setSingleStep(0.1)
        self.aspect.setSpecialValueText("as drawn")
        self.aspect.setToolTip(
            "Lock one y unit to this many x units. 'as drawn' leaves the "
            "proportions the plot already has; 1 makes the units square, "
            "which is what a Q-Q or a diagonal needs to mean anything.")
        self.aspect.valueChanged.connect(self.refresh)
        # "LOCK AXIS SCALES", NOT "ASPECT RATIO". This ties one y unit to n x
        # units and is a statement about the DATA -- what a Q-Q or a
        # diagonal needs. It is not what "save it as a square" means, and
        # calling both of them the aspect ratio is why the two were confused
        # for each other. The shape of the page is the control below.
        #
        # THE WORDS THE GRAPH'S OWN MENU USES for the same quantity, so a
        # user who met it under Axes there meets it by that name here. A
        # third name for one quantity is the same failure as one name for
        # two.
        form.addRow("lock axis scales", self.aspect)

        # THE SHAPE OF THE FIGURE, as a choice rather than a number. Asked
        # for as "change aspect ratio to graph shape and have square,
        # vertical and horizontal rectangle" -- a ratio is a number, and a
        # reader deciding how a figure sits on a page is choosing a shape.
        #
        # THE SAME VOCABULARY THE GRAPH'S OWN MENU USES, read from the one
        # table, so a figure shaped from the menu and one shaped here are
        # shaped by the same names.
        from .fast_plots import CANVAS_SHAPE_LABELS, CANVAS_SHAPES

        self.graph_shape = QComboBox()
        self.graph_shape.addItem("as drawn", "")
        for name, _ratio in CANVAS_SHAPES:
            if name == "free":
                continue
            self.graph_shape.addItem(CANVAS_SHAPE_LABELS[name], name)
        self.graph_shape.setToolTip(
            "The proportions of the saved figure. 'as drawn' keeps what is "
            "on screen. This is the shape of the PAGE; locking the axis "
            "scales above is a statement about the data.")
        self.graph_shape.currentIndexChanged.connect(self.refresh)
        form.addRow("graph shape", self.graph_shape)

        self.line_width = QDoubleSpinBox()
        self.line_width.setRange(0.0, 20.0)
        self.line_width.setDecimals(1)
        self.line_width.setSingleStep(0.5)
        self.line_width.setSpecialValueText("as drawn")
        self.line_width.setToolTip(
            "Pen width in pixels for every line. A line that reads well on "
            "screen is often too thin once the figure is a column wide.")
        self.line_width.valueChanged.connect(self.refresh)
        form.addRow("line width", self.line_width)

        self.text_px = QSpinBox()
        self.text_px.setRange(0, 72)
        self.text_px.setSpecialValueText("as drawn")
        self.text_px.setToolTip(
            "Point size for labels, ticks and the title. This is an "
            "absolute size, where 'text scale' above multiplies whatever "
            "the plot already uses.")
        self.text_px.valueChanged.connect(self.refresh)
        form.addRow("text size", self.text_px)

        self.x_title = QLineEdit()
        self.x_title.setPlaceholderText("as drawn")
        self.x_title.setToolTip(
            "The x-axis title in the saved file. The plot on screen keeps "
            "the one it has.")
        self.x_title.textChanged.connect(self.refresh)
        form.addRow("x axis title", self.x_title)

        self.y_title = QLineEdit()
        self.y_title.setPlaceholderText("as drawn")
        self.y_title.setToolTip(
            "The y-axis title in the saved file. The plot on screen keeps "
            "the one it has.")
        self.y_title.textChanged.connect(self.refresh)
        form.addRow("y axis title", self.y_title)

        # Text scale is applied to a fresh copy on every preview refresh.
        self.font_scale = QDoubleSpinBox()
        self.font_scale.setRange(0.25, 4.0)
        self.font_scale.setSingleStep(0.05)
        self.font_scale.setDecimals(2)
        self.font_scale.setValue(1.0)
        self.font_scale.setToolTip(
            "Scale labels, tick text, legends, and titles in the exported "
            "file. Use a value below 1 when reducing the page size; 1 keeps "
            "the current text size.")
        self.font_scale.valueChanged.connect(self.refresh)
        form.addRow("text scale", self.font_scale)

        size = QHBoxLayout()
        self.width = QDoubleSpinBox()
        self.width.setRange(1.0, 40.0)
        self.width.setSuffix(" in")
        self.height = QDoubleSpinBox()
        self.height.setRange(1.0, 40.0)
        self.height.setSuffix(" in")
        if self._fast:
            # The page a pyqtgraph plot writes onto is set in millimetres on
            # its OWN right-click menu (`set_export_size`), and it is read by
            # all three of its export paths. Offering a second answer in
            # inches here would give the user two controls for one quantity
            # and no way to tell which won -- so these say why instead.
            width_mm, height_mm = figure.export_size()
            self.width.setValue(round(float(width_mm) / 25.4, 2))
            self.height.setValue(round(float(height_mm or width_mm) / 25.4, 2))
            for box in (self.width, self.height):
                box.setEnabled(False)
                box.setToolTip(_SIZE_REASON)
        elif figure is not None:
            w, h = figure.get_size_inches()
            self.width.setValue(float(w))
            self.height.setValue(float(h))
        self.width.valueChanged.connect(self.refresh)
        self.height.valueChanged.connect(self.refresh)
        size.addWidget(self.width)
        size.addWidget(QLabel("×"))
        size.addWidget(self.height)
        form.addRow(_reason_label("size", _SIZE_REASON if self._fast else ""),
                    size)

        self.dpi = QSpinBox()
        self.dpi.setRange(72, 1200)
        self.dpi.setValue(300)
        self.dpi.setToolTip(
            "Dots per inch in the written file. 300 is the usual journal "
            "minimum; the screen never needs more than about 150.")
        if self._fast:
            # Vector PDF and SVG output has no DPI. Fast-plot PNG dimensions
            # come from the canvas shape, so a second resolution control would
            # be misleading.
            self.dpi.setEnabled(False)
            self.dpi.setToolTip(_RESOLUTION_REASON)
        form.addRow(_reason_label("resolution",
                                  _RESOLUTION_REASON if self._fast else ""),
                    self.dpi)

        self.format = QComboBox()
        for value, label in (FAST_PLOT_FORMATS if self._fast else FORMATS):
            self.format.addItem(label, value)
        form.addRow("format", self.format)
        layout.addLayout(form)

        self._holder = QVBoxLayout()
        layout.addLayout(self._holder, 1)

        note = QLabel("The figure on screen is not changed — these settings "
                      "apply to the file only.")
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self._save = QPushButton("Save…")
        self._save.clicked.connect(self.save)
        buttons.addButton(self._save, QDialogButtonBox.AcceptRole)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.resize(760, 760)
        self.refresh()

    # ------------------------------------------------------------- preview

    def preview(self):
        """Return the current detached preview.

        Returns
        -------
        matplotlib.figure.Figure, QPixmap, or None
            Matplotlib copy, fast-plot snapshot, or ``None`` when no preview
            can be rendered.
        """
        return self._preview

    def refresh(self, *_args):
        """Rebuild and return the preview from the original figure.

        Each refresh starts from a new copy so repeated text scaling or color
        changes do not accumulate. Fast plots delegate to
        :meth:`_refresh_fast_plot`.
        """
        from .graph_builder import _canvas_class

        if self._fast:
            return self._refresh_fast_plot()
        self._preview = style_for_file(
            copy_figure(self._source),
            ink=str(self.ink.currentData() or ""),
            background=str(self.background.currentData() or ""),
            grid=self.grid.isChecked(),
            width=float(self.width.value()), height=float(self.height.value()),
            dpi=int(self.dpi.value()),
            font_scale=float(self.font_scale.value()))
        while self._holder.count():
            item = self._holder.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        if self._preview is None:
            self._holder.addWidget(QLabel(
                "This figure cannot be previewed, so it will be written "
                "exactly as it appears on screen."))
            self._canvas = None
            return None
        self._canvas = _canvas_class()(self._preview)
        self._holder.addWidget(self._canvas)
        return self._preview

    def _extra_styling(self) -> dict:
        """The styling knobs that are only for the file, as keywords.

        A control left at its special value is left OUT rather than passed
        as zero: the render treats "no value" as "keep what the plot has",
        and passing zero would mean "make it zero".
        """
        out: dict = {}
        if self.aspect.value() > 0:
            out["aspect"] = float(self.aspect.value())
        shape = self.graph_shape.currentData()
        if shape:
            out["canvas_shape"] = str(shape)
        if self.line_width.value() > 0:
            out["line_width"] = float(self.line_width.value())
        if self.text_px.value() > 0:
            out["font_size"] = int(self.text_px.value())
        if self.x_title.text().strip():
            out["x_title"] = self.x_title.text().strip()
        if self.y_title.text().strip():
            out["y_title"] = self.y_title.text().strip()
        return out

    def _refresh_fast_plot(self):
        """Build a fast-plot preview through its styled snapshot protocol.

        The snapshot and file export use the same styling path, keeping the
        preview consistent with the saved output.
        """
        from PySide6.QtCore import Qt

        pixmap = None
        try:
            pixmap = self._source.styled_snapshot(
                760,
                ink=str(self.ink.currentData() or ""),
                background=str(self.background.currentData() or ""),
                grid=self.grid.isChecked(),
                **self._extra_styling())
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not preview the plot", exc_info=True)
        self._clear_holder()
        if pixmap is None:
            self._holder.addWidget(QLabel(
                "This plot has nothing drawn on it yet, so there is nothing "
                "to preview or to save."))
            self._save.setEnabled(False)
            return None
        self._save.setEnabled(True)
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        label.setPixmap(pixmap)
        self._holder.addWidget(label)
        # The pixmap IS the preview; there is no figure object behind it.
        self._preview = pixmap
        return pixmap

    def _clear_holder(self) -> None:
        while self._holder.count():
            item = self._holder.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    # ---------------------------------------------------------------- save

    def save(self, path: str = "") -> str:
        """Write the figure using the current export settings.

        Parameters
        ----------
        path : str, optional
            Destination path. When omitted, a file chooser is displayed.

        Returns
        -------
        str
            Written path, or an empty string when the chooser is cancelled or
            no figure can be saved.
        """
        chosen = str(path or "")
        suffix = str(self.format.currentData() or "png")
        if not chosen:
            chosen, _filter = QFileDialog.getSaveFileName(
                self, "Save figure", f"figure.{suffix}",
                f"{dict(FORMATS).get(suffix, suffix)} (*.{suffix})")
        if not chosen:
            return ""
        if self._fast:
            try:
                written = self._source.export_styled(
                    chosen,
                    ink=str(self.ink.currentData() or ""),
                    background=str(self.background.currentData() or ""),
                    grid=self.grid.isChecked(),
                    **self._extra_styling())
            except Exception:                                # noqa: BLE001
                LOG.debug("could not save the plot", exc_info=True)
                return ""
            if not written:
                return ""
            self.accept()
            return str(written)
        target = self._preview if self._preview is not None else self._source
        if target is None:
            return ""
        try:
            # NOT `plot.save_figure`, and deliberately (108 point 6). Every
            # decision that writer makes -- the format, the DPI, the page
            # colour -- the user has just made in this dialog and is looking
            # at in the preview. Overriding any of them here would write
            # something other than what was previewed, which is the one thing
            # this window promises not to do.
            target.savefig(chosen, dpi=int(self.dpi.value()),
                           bbox_inches="tight",
                           facecolor=target.patch.get_facecolor(),
                           transparent=not self.background.currentData())
        except Exception as exc:                             # noqa: BLE001
            LOG.debug("could not save the figure", exc_info=True)
            return ""
        self.accept()
        return chosen
