"""Preview and save a separately styled copy of a Matplotlib figure.

Export styling is applied to a detached copy, leaving the figure displayed in
spaCR unchanged. Figures that cannot be copied still use the ordinary save
path without a styled preview.
"""
import logging
from typing import Optional

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

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
    """Whether ``figure`` is one of this application's pyqtgraph plots.

    BY CAPABILITY, not by class. The dialog needs exactly two things --
    a styled preview and a styled write -- and asking for those is what keeps
    a new plot class working here without being added to a list somebody has
    to remember.
    """
    return (figure is not None
            and callable(getattr(figure, "styled_snapshot", None))
            and callable(getattr(figure, "export_styled", None)))


def copy_figure(figure):
    """A detached copy of ``figure``, or ``None``.

    Pickled and unpickled: the same round trip `FigureQueue` spills through,
    so anything the queue can evict and restore can be previewed. A figure
    that will not pickle -- one holding a live canvas or a closure -- comes
    back None, and the caller offers the plain save instead of a broken
    preview.
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
    """Apply the FILE's styling to ``figure``. Returns it.

    Deliberately small and explicit rather than routed through the live
    restyle path: this runs on a copy that nothing else will ever see again,
    so it has no preferences to respect and no theme to follow.
    """
    if figure is None:
        return None
    if width and height:
        figure.set_size_inches(float(width), float(height))
    if dpi:
        figure.set_dpi(int(dpi))
    # TEXT SIZED FOR THE PAGE (187 D2). The reported fault was "ginormous
    # text" on an exported figure, and the cause on the pyqtgraph side was a
    # device scale (fixed separately) -- but a matplotlib figure resized for
    # a journal column has the opposite problem: the axes shrink and the
    # labels do not, so a figure drawn at 10 inches and saved at 3.4 is all
    # text. Scaling every text artist by the same factor is the one control
    # that answers it, and it is applied to the artists rather than to
    # rcParams because rcParams only reach an artist when it is CREATED and
    # this figure already exists.
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
        # UNTICKING "draw a grid" DREW A GRID. matplotlib warns on it --
        # "First parameter to grid() is false, but line properties are
        # supplied. The grid will be enabled." -- and does exactly that:
        # measured, `grid(False, linewidth=..., alpha=...)` leaves gridOn
        # True. The line properties only mean anything when the grid is on,
        # so they are passed only then.
        if grid:
            axes.grid(True, which="major", linewidth=0.4, alpha=0.35)
        else:
            axes.grid(False)
    return figure


class SaveFigureDialog(QDialog):
    """Style, preview, then write -- leaving the on-screen figure alone."""

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

        # TEXT SIZE, LIVE (187 D2). "in save styled there should be
        # comprehensive figure setting that change the figure live" -- and
        # text is the first candidate, because the report that prompted it
        # was a figure whose text was wrong for the page. Every control here
        # already redraws the preview; that is what makes it a preview
        # rather than a form.
        self.font_scale = QDoubleSpinBox()
        self.font_scale.setRange(0.25, 4.0)
        self.font_scale.setSingleStep(0.05)
        self.font_scale.setDecimals(2)
        self.font_scale.setValue(1.0)
        self.font_scale.setToolTip(
            "Scale every label, tick and title by this factor. A figure "
            "drawn at 10 inches and saved at 3.4 is all text unless the "
            "type comes down with it.")
        self.font_scale.valueChanged.connect(self.refresh)
        form.addRow("text size", self.font_scale)

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
            # A pyqtgraph PDF and SVG are true vector -- they have no dpi at
            # all -- and its PNG is sized in pixels by the same canvas shape.
            # Instruction 106: disabled and SAYING why.
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
        return self._preview

    def refresh(self, *_args):
        """Rebuild the preview from a fresh copy of the original.

        FROM THE ORIGINAL EVERY TIME, never from the last preview: styling a
        styled copy compounds, so moving the ink back to "as on screen" would
        not undo the first change.
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

    def _refresh_fast_plot(self):
        """The preview for a pyqtgraph plot: THE SAME RENDER THE FILE GETS.

        Not an approximation of it. `styled_snapshot` and `export_styled` wear
        the same styling through the same context manager, so a preview that
        looked right and a file that did not would take a change to both to
        produce.
        """
        from PySide6.QtCore import Qt

        pixmap = None
        try:
            pixmap = self._source.styled_snapshot(
                760,
                ink=str(self.ink.currentData() or ""),
                background=str(self.background.currentData() or ""),
                grid=self.grid.isChecked())
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
        """Write the previewed figure. Returns the path, or ``""``."""
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
                    grid=self.grid.isChecked())
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
            target.savefig(chosen, dpi=int(self.dpi.value()),
                           bbox_inches="tight",
                           facecolor=target.patch.get_facecolor(),
                           transparent=not self.background.currentData())
        except Exception as exc:                             # noqa: BLE001
            LOG.debug("could not save the figure", exc_info=True)
            return ""
        self.accept()
        return chosen
