"""Preview and save a figure styled for the file rather than for the screen.

The figure displayed in spaCR is never changed. A Matplotlib figure is copied
before it is styled; a pyqtgraph plot cannot be copied safely, so it is styled
only for the length of one offscreen render and put back afterwards. Figures
that can be neither copied nor rendered still use the ordinary save path
without a styled preview.

The dialog offers what belongs to the FILE -- its background, the colour and
width of its lines, the colour of its text, the shape of its page, and what
kind of file it is. Everything else a figure can be told belongs to the PLOT
and lives on the plot's own right-click menu, where it reaches the screen and
every export at once; a value inherited from there is shown here rather than
offered a second time.
"""
import logging
from typing import Optional

from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QPushButton,
                               QSpinBox, QVBoxLayout, QWidget)

LOG = logging.getLogger("spacr.qt.save_figure")

#: Why the page size is not editable here for a pyqtgraph plot.
_SIZE_REASON = ("set on the plot's right-click menu, under Canvas: it is one "
                "page size, read by every export this plot makes")

#: Why a vector file has no resolution to set. PDF and SVG carry paths and
#: text rather than pixels, so the page size is the whole answer.
_RESOLUTION_REASON = ("PDF and SVG are true vector: the page size is the "
                      "whole answer and there is no resolution to set")

#: What the resolution does when the file IS pixels, in one sentence. The
#: resolution decides the count and the graph shape decides the proportion,
#: which is the order a journal asking for 300 dpi expects.
_PIXELS_REASON = ("pixels across = page width in inches × resolution; the "
                  "graph shape decides the height")

#: Said when the plot really has nothing on it. Reserved for that case: a
#: render that RAISED is a different sentence and gets one.
EMPTY_PLOT = ("This plot has nothing drawn on it yet, so there is nothing "
              "to preview or to save.")

#: Said when the render raised. The figure already in the preview is left
#: where it is, so this says what it is rather than letting it be read as
#: the answer to the settings just chosen.
PREVIEW_FAILED = ("These settings could not be drawn, so nothing would be "
                  "written either. The figure shown is the last one that "
                  "could be drawn, not what these settings would give —")

#: The same, with no earlier figure to keep.
PREVIEW_FAILED_ALONE = ("These settings could not be drawn, so nothing "
                        "would be written either —")

#: Said when the write itself raised, rather than the preview.
SAVE_FAILED = "The file could not be written —"


def _with_reason(text: str, reason: str = "") -> str:
    """Compose a value or a name with the sentence that explains it.

    One composer, so a label and the note beside it are punctuated the same
    way and a reader meets one form rather than two.
    """
    if not reason:
        return text
    return f"{text} — {reason}" if text else reason


def _reason_label(text: str, reason: str = ""):
    """Return a form label that displays why its control is disabled.

    The reason appears directly in the label and as a tooltip, so it remains
    available when a disabled widget does not receive hover events.
    """
    from PySide6.QtWidgets import QLabel

    if not reason:
        return QLabel(text)
    label = QLabel(_with_reason(text, reason))
    label.setEnabled(False)
    label.setToolTip(reason)
    label.setWordWrap(True)
    return label


def _quiet_note(text: str = "", reason: str = ""):
    """A dimmed note that carries a VALUE first and its explanation after.

    The row's label names the setting and the control shows what it is set
    to; this is where the sentence about where that value comes from goes,
    so the number is what a reader about to save sees first. Dimmed rather
    than hidden, because the explanation is worth reading once and is not
    worth competing with the number for attention every time after that.
    """
    label = _reason_label(text, reason)
    label.setEnabled(False)
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

#: The formats that hold paths rather than pixels, and so have no resolution.
VECTOR_FORMATS = ("pdf", "svg")

#: Ink for the TEXT in the file: the title, the axis labels, the tick
#: numbers and the legend. "As drawn" is first because it is what the
#: previous behaviour did, and a dialog that silently changed the default
#: would restyle every save a user made out of habit.
INKS = (("", "as drawn"),
        ("#231F20", "black — for paper"),
        ("#FFFFFF", "white — for a dark slide"))

#: Ink for the LINES: the plotted curves, the reference lines, the axis
#: spines and the tick marks. The same three answers as the text, because a
#: figure going onto paper wants both and asking for them in two different
#: vocabularies is how one of them gets forgotten.
LINE_INKS = (("", "as drawn"),
             ("#231F20", "black — for paper"),
             ("#FFFFFF", "white — for a dark slide"))

#: The page behind everything.
BACKGROUNDS = (("", "transparent"), ("#FFFFFF", "white"), ("#000000", "black"))

#: Marks the entry that opens a colour chooser rather than being a colour.
#: Not a valid colour string, so a stale selection cannot be mistaken for one.
_CHOOSE = "\0choose"

#: What that entry is called.
_CHOOSE_LABEL = "choose a colour…"


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
                   grid: Optional[bool] = None, width: float = 0.0,
                   height: float = 0.0, dpi: int = 0,
                   font_scale: float = 0.0, text_colour: str = "",
                   line_colour: str = ""):
    """Apply export-only styling to a Matplotlib figure.

    Parameters
    ----------
    figure : matplotlib.figure.Figure or None
        Figure copy to modify. ``None`` is accepted for preview fallbacks.
    ink : str, optional
        Color applied to titles, labels, ticks, spines, legends, and text --
        text and lines together, which is what a paper-or-slide preset
        means. An empty string preserves the existing colors.
    text_colour : str, optional
        Color for TEXT only: the title, the axis labels, the tick numbers
        and the legend. Overrides ``ink`` for that half.
    line_colour : str, optional
        Color for LINES only: the spines and the tick marks. Overrides
        ``ink`` for that half.
    background : str, optional
        Figure and axes background color. An empty string makes both
        backgrounds transparent.
    grid : bool or None, default None
        Draw major grid lines when ``True`` and disable them when ``False``.
        ``None`` leaves the figure's own grid alone, which is what a caller
        with no grid control of its own wants: a default of ``False`` turns
        off a grid the figure was drawn with and nobody asked about.
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
    text_ink = text_colour or ink
    line_ink = line_colour or ink
    for axes in figure.axes:
        if background:
            axes.set_facecolor(background)
        else:
            axes.patch.set_alpha(0.0)
        if text_ink:
            axes.title.set_color(text_ink)
            axes.xaxis.label.set_color(text_ink)
            axes.yaxis.label.set_color(text_ink)
            # THE NUMBERS BESIDE THE TICKS ARE TEXT and the little dashes are
            # lines, so the two halves of `tick_params` follow two different
            # controls. One call with both would tie them together again.
            axes.tick_params(labelcolor=text_ink, which="both")
            legend = axes.get_legend()
            if legend is not None:
                for text in legend.get_texts():
                    text.set_color(text_ink)
            for text in axes.texts:
                text.set_color(text_ink)
        if line_ink:
            axes.tick_params(color=line_ink, which="both")
            for spine in axes.spines.values():
                spine.set_edgecolor(line_ink)
        # Matplotlib enables the grid when line properties accompany
        # ``grid(False)``. Supply styling arguments only for the enabled case.
        if grid:
            axes.grid(True, which="major", linewidth=0.4, alpha=0.35)
        elif grid is not None:
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

    WHAT IT OFFERS, AND WHY IT IS SHORT. Four settings change how the FILE
    looks -- the background, the lines' colour and width, the text colour,
    and the shape of the page -- and three more say what the file IS: its
    format, its resolution and its page size. Everything else a figure can
    be told is a property of the PLOT and lives on the plot's own right-click
    menu, where it applies to the screen and to every export at once. A
    second copy of those controls here would be a second answer to one
    question with no way to tell which won.

    A SETTING INHERITED FROM THE PLOT IS SHOWN, NOT EXPLAINED. The page size
    a pyqtgraph plot writes onto is set on that menu; this dialog displays
    the size it will get and keeps the sentence about where it comes from as
    a quieter note beside the number.
    """

    #: How wide the preview is rendered, in pixels.
    PREVIEW_PX = 760

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
        #: Guards the colour combos while a cancelled chooser is put back.
        self._picking = False
        #: Height over width of the figure as it was handed in, for a
        #: Matplotlib figure whose page this dialog reshapes directly.
        self._drawn_ratio: Optional[float] = None

        layout = QVBoxLayout(self)
        form = QFormLayout()

        # ------------------------------------------- the four that are the file's
        self.background = self._colour_box(
            BACKGROUNDS,
            "The page behind the figure. Transparent takes whatever the "
            "figure is placed on, which is what a slide usually wants.")
        form.addRow("background colour", self.background)

        self.line_colour = self._colour_box(
            LINE_INKS,
            "Every line in the file: the plotted curves, the reference "
            "lines, the axis spines and the tick marks. The numbers beside "
            "the ticks are text and follow the text colour.")
        form.addRow("line colour", self.line_colour)

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

        # NAMED `ink` IN THE CODE, "text colour" ON THE ROW. The attribute is
        # what every caller and test already reaches for; the label is the
        # word the maintainer used and the word the row beside it uses.
        self.ink = self._colour_box(
            INKS,
            "Every piece of text in the file: the title, the axis labels, "
            "the numbers beside the ticks and the legend.")
        form.addRow("text colour", self.ink)

        # THE SHAPE OF THE FIGURE, as a choice rather than a number. A ratio
        # is a number, and a reader deciding how a figure sits on a page is
        # choosing a shape.
        #
        # THE SAME VOCABULARY THE GRAPH'S OWN MENU USES, read from the one
        # table, so a figure shaped from the menu and one shaped here are
        # shaped by the same names. "Lock axis scales" -- one y unit drawn as
        # n x units -- is a statement about the DATA and lives on that menu
        # under Axes; it is not what "save it as a square" means.
        try:
            from .fast_plots import CANVAS_SHAPE_LABELS, CANVAS_SHAPES
        except Exception:                                    # noqa: BLE001
            CANVAS_SHAPES, CANVAS_SHAPE_LABELS = (), {}
        self._shape_ratios = {name: ratio for name, ratio in CANVAS_SHAPES}

        self.graph_shape = QComboBox()
        self.graph_shape.addItem("as drawn", "")
        for name, _ratio in CANVAS_SHAPES:
            if name == "free":
                continue
            self.graph_shape.addItem(CANVAS_SHAPE_LABELS[name], name)
        self.graph_shape.setToolTip(
            "The proportions of the saved figure. 'as drawn' keeps what is "
            "on screen. This is the shape of the PAGE; the plot's own menu "
            "has the axis lock, which is a statement about the data.")
        self.graph_shape.currentIndexChanged.connect(self._shape_changed)
        form.addRow("graph shape", self.graph_shape)

        # ---------------------------------------------- what the file IS
        self.format = QComboBox()
        for value, label in (FAST_PLOT_FORMATS if self._fast else FORMATS):
            self.format.addItem(label, value)
        self.format.currentIndexChanged.connect(self._format_changed)
        form.addRow("format", self.format)

        # A RASTER EXPORT IS WHAT A RESOLUTION IS FOR. It follows the FORMAT
        # and not the kind of plot: a journal asking for 300 dpi is asking
        # about the PNG, and greying the one control that decides how big
        # that file really is takes the answer away.
        self.dpi = QSpinBox()
        self.dpi.setRange(72, 1200)
        self.dpi.setValue(300)
        self.dpi.valueChanged.connect(self._page_changed)
        self._resolution_note = _quiet_note()
        resolution = QHBoxLayout()
        resolution.addWidget(self.dpi)
        resolution.addWidget(self._resolution_note, 1)
        form.addRow("resolution", resolution)

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
            # and no way to tell which won -- so these SHOW it instead.
            for box in (self.width, self.height):
                box.setEnabled(False)
                box.setToolTip(_SIZE_REASON)
        elif figure is not None:
            w, h = figure.get_size_inches()
            self.width.setValue(float(w))
            self.height.setValue(float(h))
            #: The proportion the figure was drawn at, so "as drawn" has
            #: something to go back to after a shape has been chosen.
            self._drawn_ratio = float(h) / float(w) if w else None
        self.width.valueChanged.connect(self.refresh)
        self.height.valueChanged.connect(self.refresh)
        self._size_note = _quiet_note()
        size.addWidget(self.width)
        size.addWidget(QLabel("×"))
        size.addWidget(self.height)
        size.addWidget(self._size_note, 1)
        form.addRow("size", size)
        layout.addLayout(form)

        self._holder = QVBoxLayout()
        layout.addLayout(self._holder, 1)

        # WHERE A REFUSAL IS SAID OUT LOUD. A preview or a save that fails
        # writes its reason here; an empty label is hidden, so the dialog
        # gains a line only when there is something to read.
        self._trouble = QLabel()
        self._trouble.setWordWrap(True)
        self._trouble.setVisible(False)
        layout.addWidget(self._trouble)

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
        self._format_changed()

    # -------------------------------------------------------------- colours

    def _colour_box(self, choices, tooltip: str) -> QComboBox:
        """A colour combo: the presets, then a chooser for anything else.

        The presets are the two answers a figure leaving the application
        usually needs -- black on paper, white on a dark slide -- and the
        chooser is there so the dialog is not weaker than the plot's own
        menu, which has had a full picker all along.
        """
        box = QComboBox()
        for value, label in choices:
            box.addItem(label, value)
        box.addItem(_CHOOSE_LABEL, _CHOOSE)
        box.setToolTip(tooltip)
        # THE CHOOSER RUNS FIRST. Connected before the refresh so a chosen
        # colour is already in the combo by the time the preview is rebuilt;
        # the other order previews the sentinel and then the colour.
        box.currentIndexChanged.connect(
            lambda _index, which=box: self._resolve_choice(which))
        box.currentIndexChanged.connect(self.refresh)
        return box

    def _resolve_choice(self, box: QComboBox) -> None:
        """Turn a "choose a colour…" selection into a colour, or undo it."""
        if self._picking or box.currentData() != _CHOOSE:
            return
        from .colour_picker import pick_colour

        self._picking = True
        try:
            chosen = pick_colour(self, "#FFFFFF", "Colour")
            index = 0
            if chosen.isValid():
                name = chosen.name()
                index = box.findData(name)
                if index < 0:
                    # Before the chooser, so the chooser stays last.
                    index = box.count() - 1
                    box.insertItem(index, name, name)
            box.setCurrentIndex(index)
        finally:
            self._picking = False

    @staticmethod
    def _colour_of(box: QComboBox) -> str:
        """The colour a combo holds, or an empty string for "as drawn"."""
        value = box.currentData()
        return "" if not value or value == _CHOOSE else str(value)

    # ---------------------------------------------------------- the file rows

    def _format_changed(self, *_args) -> None:
        """Light the resolution for a raster format and grey it for vector.

        THE FORMAT DECIDES, NOT THE KIND OF PLOT. Vector output has no
        resolution to set; a PNG's resolution is the number that decides how
        big the file really is, and that is true of a pyqtgraph plot exactly
        as it is of a Matplotlib one.
        """
        vector = self._suffix() in VECTOR_FORMATS
        self.dpi.setEnabled(not vector)
        self.dpi.setToolTip(
            _RESOLUTION_REASON if vector else
            "Dots per inch in the written file. 300 is the usual journal "
            "minimum; the screen never needs more than about 150.")
        self._page_changed()

    def _suffix(self) -> str:
        return str(self.format.currentData() or "png")

    def _shape_ratio(self) -> Optional[float]:
        """Height over width for the chosen shape, or None for "as drawn"."""
        name = str(self.graph_shape.currentData() or "")
        return self._shape_ratios.get(name) if name else None

    def _page_mm(self) -> tuple:
        """``(width, height)`` of the page in millimetres, or ``(None, None)``.

        Only a pyqtgraph plot has a page measured in millimetres. The graph
        shape chosen here is applied to it, because that is the shape the
        file is about to be written at and showing the unshaped height would
        be showing a number the save will not use.
        """
        if not self._fast:
            return None, None
        try:
            width_mm, height_mm = self._source.export_size()
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not read the plot's page size", exc_info=True)
            return None, None
        ratio = self._shape_ratio()
        if ratio is not None:
            height_mm = float(width_mm) * float(ratio)
        return float(width_mm), (None if height_mm is None
                                 else float(height_mm))

    def _shape_changed(self, *_args) -> None:
        """Put the chosen shape onto the page, then rebuild everything.

        A pyqtgraph plot's page is millimetres and :meth:`_show_the_page`
        recomputes it. A Matplotlib figure's page is the inches in the size
        row, so the shape is applied by writing the height -- which also
        keeps rule 3, because the number shown is then the number used.
        """
        ratio = self._shape_ratio()
        if not self._fast and self._drawn_ratio is not None:
            proportion = self._drawn_ratio if ratio is None else float(ratio)
            blocked = self.height.blockSignals(True)
            try:
                self.height.setValue(
                    round(float(self.width.value()) * proportion, 2))
            finally:
                self.height.blockSignals(blocked)
        self._page_changed()

    def _page_changed(self, *_args) -> None:
        """Re-read the page and the pixel count, then rebuild the preview."""
        self._show_the_page()
        self.refresh()

    def _show_the_page(self) -> None:
        """Put the page size and the pixel count on the rows that show them.

        RULE 3: the row shows the VALUE. "Set on the plot's right-click menu"
        explains where a number lives and is no substitute for the number,
        so the millimetres and the pixels are written out and the sentence
        about where they come from is the quieter note beside them.
        """
        width_mm, height_mm = self._page_mm()
        if width_mm is not None:
            for box, inches in ((self.width, width_mm / 25.4),
                                (self.height,
                                 (height_mm if height_mm else width_mm) / 25.4)):
                # Written with the handler blocked: these boxes only REPORT
                # the plot's page, and letting them re-enter the refresh that
                # is about to run renders the preview twice per keystroke.
                blocked = box.blockSignals(True)
                try:
                    box.setValue(round(inches, 2))
                finally:
                    box.blockSignals(blocked)
            shown = (f"{width_mm:g} × {height_mm:g} mm" if height_mm
                     else f"{width_mm:g} mm wide, height follows the plot")
            self._size_note.setText(_with_reason(shown, _SIZE_REASON))
            self._size_note.setToolTip(_SIZE_REASON)
        else:
            self._size_note.setText("")
        if self._suffix() in VECTOR_FORMATS:
            self._resolution_note.setText(_RESOLUTION_REASON)
            self._resolution_note.setToolTip(_RESOLUTION_REASON)
            return
        pixels = self._raster_pixels()
        if not pixels:
            shown = ""
        elif pixels[1] is None:
            shown = f"{pixels[0]} pixels across, height follows the plot"
        else:
            shown = f"{pixels[0]} × {pixels[1]} pixels"
        self._resolution_note.setText(_with_reason(shown, _PIXELS_REASON))
        self._resolution_note.setToolTip(_PIXELS_REASON)

    def _raster_pixels(self) -> Optional[tuple]:
        """``(width, height)`` in pixels, with height None when it follows.

        ONLY FOR A PLOT WHOSE PAGE IS A KNOWN SIZE. A Matplotlib figure is
        written with ``bbox_inches="tight"``, which crops the page to the
        ink -- so a count printed for one would be an upper bound presented
        as a measurement, and a wrong number shown confidently is what rule
        3 is trying to get away from.

        The WIDTH is always known once the page is: it is the resolution
        times the page width, and it is the number that says how big the
        file will be. The height is known only once a shape has been chosen,
        so an unshaped page says so rather than guessing one.
        """
        width_mm, height_mm = self._page_mm()
        if width_mm is None:
            return None
        width = max(1, int(round(width_mm / 25.4 * int(self.dpi.value()))))
        ratio = self._shape_ratio()
        if ratio is None and width_mm and height_mm:
            ratio = float(height_mm) / float(width_mm)
        if ratio is None:
            return width, None
        return width, max(1, int(round(width * float(ratio))))

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

    def _say(self, trouble: str) -> None:
        """Show a refusal, or clear the line when there is nothing to say."""
        self._trouble.setText(trouble)
        self._trouble.setToolTip(trouble)
        self._trouble.setVisible(bool(trouble))

    @staticmethod
    def _why(exc: BaseException) -> str:
        """The reason a render or a write refused, in a readable line."""
        reason = str(exc).strip()
        return f"{type(exc).__name__}: {reason}" if reason else type(exc).__name__

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
            text_colour=self._colour_of(self.ink),
            line_colour=self._colour_of(self.line_colour),
            background=self._colour_of(self.background),
            width=float(self.width.value()), height=float(self.height.value()),
            dpi=int(self.dpi.value()))
        self._clear_holder()
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
        shape = self.graph_shape.currentData()
        if shape:
            out["canvas_shape"] = str(shape)
        if self.line_width.value() > 0:
            out["line_width"] = float(self.line_width.value())
        line_ink = self._colour_of(self.line_colour)
        if line_ink:
            out["line_colour"] = line_ink
        text_ink = self._colour_of(self.ink)
        if text_ink:
            out["text_colour"] = text_ink
        return out

    def _for_the_file(self) -> dict:
        """The styling, plus what the FILE is: its page and its resolution.

        The resolution is not styling -- it decides how many pixels the file
        has, not how the figure looks -- so it is kept out of
        :meth:`_extra_styling` and added here, where the render and the
        write both read it. It is passed only for a format that has one:
        vector output ignores a resolution, and sending one anyway would
        make a PDF look as though it had been given a choice it cannot use.
        """
        out = self._extra_styling()
        if self.dpi.isEnabled():
            out["dpi"] = int(self.dpi.value())
        return out

    def _refresh_fast_plot(self):
        """Build a fast-plot preview through its styled snapshot protocol.

        The snapshot and file export use the same styling path, keeping the
        preview consistent with the saved output.

        A PREVIEW THAT FAILS SAYS WHY, and never calls a drawn plot empty.
        "Nothing drawn" is one reason a preview can be absent; a render that
        RAISED is another, and the two are not the same sentence. The
        exception used to go to a debug log nobody reads and the dialog then
        blamed the plot for being empty -- a full volcano reported as a blank
        one, with the real reason discarded on the way.
        """
        from PySide6.QtCore import Qt

        pixmap, failure = None, ""
        try:
            pixmap = self._source.styled_snapshot(
                self.PREVIEW_PX,
                background=self._colour_of(self.background),
                **self._for_the_file())
        except Exception as exc:                             # noqa: BLE001
            failure = self._why(exc)
            LOG.debug("could not preview the plot", exc_info=True)
        if pixmap is not None:
            self._clear_holder()
            label = QLabel()
            label.setAlignment(Qt.AlignCenter)
            label.setPixmap(pixmap)
            self._holder.addWidget(label)
            # The pixmap IS the preview; there is no figure object behind it.
            self._preview = pixmap
            self._save.setEnabled(True)
            self._say("")
            return pixmap
        self._save.setEnabled(False)
        if failure:
            # KEEP THE FIGURE. The holder is NOT cleared: what is in it is
            # the last drawing that worked, and the note says exactly that
            # so it cannot be read as the answer to the settings just made.
            self._say(f"{PREVIEW_FAILED} {failure}" if self._preview is not None
                      else f"{PREVIEW_FAILED_ALONE} {failure}")
            return None
        self._clear_holder()
        self._preview = None
        self._holder.addWidget(QLabel(EMPTY_PLOT))
        self._say("")
        return None

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
        suffix = self._suffix()
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
                    background=self._colour_of(self.background),
                    **self._for_the_file())
            except Exception as exc:                         # noqa: BLE001
                LOG.debug("could not save the plot", exc_info=True)
                self._say(f"{SAVE_FAILED} {self._why(exc)}")
                return ""
            if not written:
                self._say(f"{SAVE_FAILED} the plot wrote no file.")
                return ""
            self.accept()
            return str(written)
        target = self._preview if self._preview is not None else self._source
        if target is None:
            self._say(f"{SAVE_FAILED} there is no figure to write.")
            return ""
        try:
            # NOT `plot.save_figure`, and deliberately. Every decision that
            # writer makes -- the format, the DPI, the page colour -- the
            # user has just made in this dialog and is looking at in the
            # preview. Overriding any of them here would write something
            # other than what was previewed, which is the one thing this
            # window promises not to do.
            target.savefig(chosen, dpi=int(self.dpi.value()),
                           bbox_inches="tight",
                           facecolor=target.patch.get_facecolor(),
                           transparent=not self._colour_of(self.background))
        except Exception as exc:                             # noqa: BLE001
            LOG.debug("could not save the figure", exc_info=True)
            self._say(f"{SAVE_FAILED} {self._why(exc)}")
            return ""
        self.accept()
        return chosen
