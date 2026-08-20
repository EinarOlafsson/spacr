"""Style a figure FOR THE FILE, see it, then write it (instruction 178 C.2).

    "when right clicking and saving a figure (matplotlib or pyqt6graph) the
    user should be able to change all of theis for the saved graph, get a
    preview then save."

THE FIGURE ON SCREEN IS NOT TOUCHED, and that is the whole design. A saved
figure is for paper and the one on screen is for the screen (instruction
150); a save that restyled the live figure would change what the user is
looking at as a side effect of writing a file, and they would have to undo it
afterwards to carry on reading.

So the preview is a COPY. The copy is made by pickling the figure -- the same
mechanism `FigureQueue` already uses to spill figures to disk, so a figure
that can be evicted and restored can be previewed -- and every control here
acts on the copy alone.
"""
import logging
from typing import Optional

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

LOG = logging.getLogger("spacr.qt.save_figure")

#: What the file can be written as.
FORMATS = (("png", "PNG image"), ("pdf", "PDF document"),
           ("svg", "SVG image"), ("tiff", "TIFF image"))

#: Ink choices for the file. "As on screen" is first because it is what the
#: previous behaviour did, and a dialog that silently changed the default
#: would restyle every save a user made out of habit.
INKS = (("", "as on screen"),
        ("#231F20", "black ink on white — for paper"),
        ("#FFFFFF", "white ink on transparent — for a dark slide"))


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
                   height: float = 0.0, dpi: int = 0):
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
        axes.grid(bool(grid), which="major", linewidth=0.4, alpha=0.35)
    return figure


class SaveFigureDialog(QDialog):
    """Style, preview, then write -- leaving the on-screen figure alone."""

    def __init__(self, figure, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Save figure")
        self._source = figure
        self._preview = None
        self._canvas = None

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

        size = QHBoxLayout()
        self.width = QDoubleSpinBox()
        self.width.setRange(1.0, 40.0)
        self.width.setSuffix(" in")
        self.height = QDoubleSpinBox()
        self.height.setRange(1.0, 40.0)
        self.height.setSuffix(" in")
        if figure is not None:
            w, h = figure.get_size_inches()
            self.width.setValue(float(w))
            self.height.setValue(float(h))
        self.width.valueChanged.connect(self.refresh)
        self.height.valueChanged.connect(self.refresh)
        size.addWidget(self.width)
        size.addWidget(QLabel("×"))
        size.addWidget(self.height)
        form.addRow("size", size)

        self.dpi = QSpinBox()
        self.dpi.setRange(72, 1200)
        self.dpi.setValue(300)
        self.dpi.setToolTip(
            "Dots per inch in the written file. 300 is the usual journal "
            "minimum; the screen never needs more than about 150.")
        form.addRow("resolution", self.dpi)

        self.format = QComboBox()
        for value, label in FORMATS:
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

        self._preview = style_for_file(
            copy_figure(self._source),
            ink=str(self.ink.currentData() or ""),
            background=str(self.background.currentData() or ""),
            grid=self.grid.isChecked(),
            width=float(self.width.value()), height=float(self.height.value()),
            dpi=int(self.dpi.value()))
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
