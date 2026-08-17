"""Interactive regression plots drawn by Qt, not by matplotlib.

WHY THIS EXISTS

matplotlib redraws every artist, in Python, on every frame. On the screen's
volcano -- 1,215 points scattered once per LOPIT compartment, with a 27-entry
legend -- that is ~115 ms per redraw, and it is paid again for every pan, every
zoom, and every style change. No amount of debouncing, threading or resolution
capping removes it, because the cost is text layout and marker geometry rather
than pixels.

pyqtgraph draws into a QGraphicsScene. Pan, zoom and hover cost NOTHING, because
the scene is composited by Qt rather than re-rendered by Python; a log-axis
toggle is 4.7 ms against matplotlib's 115 ms; recolouring every point is 45 ms.
The same reason the 3D UMAP viewer can spin 10,000 points with edges while a
flat scatter plot stutters.

THE SPLIT

    on screen  -> pyqtgraph, because the user interacts with it
    on disk    -> matplotlib, because it still makes the better vector page

Publication figures are unchanged. This is only what the application shows.

Every plot here takes a DataFrame and returns a widget. They are deliberately
free of spaCR imports so they can be tested, and reused, on their own.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

try:  # pragma: no cover - exercised by the import guard test
    import pyqtgraph as pg
    from pyqtgraph import ScatterPlotItem
    HAVE_PYQTGRAPH = True
except Exception:  # pragma: no cover - pyqtgraph is optional
    pg = None
    ScatterPlotItem = object
    HAVE_PYQTGRAPH = False


class _Absorbs:
    """Answers any call, any attribute, with something harmless.

    Stands in for BOTH the pyqtgraph module and a PlotWidget when pyqtgraph
    is not installed, so the thirty ``pg.`` calls and the forty
    ``self.plot.`` calls in this file do not each need a guard. A guard per
    call site is how one gets missed on the path nobody tested, and the
    crash comes back somewhere new.

    Every attribute is a callable returning an empty list, because the two
    things callers do with a result are ignore it and iterate it
    (``listDataItems``, ``actions``). Returning None satisfies the first and
    raises on the second.
    """

    def __getattr__(self, _name):
        return self

    def __call__(self, *args, **kwargs):
        return self

    # Chains have to survive whole, not one link at a time. The subclasses
    # run `self.plot.scene().sigMouseClicked.connect(...)` in their own
    # __init__ -- three links -- so returning a plain [] from the first call
    # only moves the AttributeError one step along.
    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def __bool__(self):
        # `if self._highlight:` must read as "nothing is drawn", which is
        # true, rather than as a live artist to remove.
        return False

    def __repr__(self):
        return "<pyqtgraph absent>"


if not HAVE_PYQTGRAPH:  # pragma: no cover - exercised by the absence test
    # THE MODULE TOO, not only the widget. Thirty call sites in this file go
    # through `pg.` -- mkBrush, mkPen, ScatterPlotItem, InfiniteLine -- and
    # `pg = None` turns every one into an AttributeError the moment a table
    # arrives. The panel would then BUILD and die on its first redraw, which
    # is a worse failure than the original: the app looks fine until the user
    # loads data.
    pg = _Absorbs()


#: What the user is told, and what they can do about it. Names the EXTRA the
#: way NAPARI_MISSING_MESSAGE does, rather than the bare distribution: a
#: `pip install pyqtgraph` into an environment installed from an extra is how
#: people end up with a package the next upgrade quietly removes again.
PYQTGRAPH_MISSING_MESSAGE = (
    "Interactive plots need pyqtgraph.\n\n"
    "Install it with  pip install 'spacr[qt]'  and reopen this module.\n\n"
    "Everything else works without it: the run still produces every figure, "
    "and they appear on the grid above the console.")

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget,
)

#: Colour-blind-safe qualitative order. A screen's categories are nominal, so a
#: sequential map would imply a ranking that is not there.
PALETTE = (
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860",
    "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD", "#4878CF", "#EE854A",
    "#6ACC64", "#D65F5F", "#956CB4", "#8C613C", "#DC7EC0", "#797979",
    "#D5BB67", "#82C6E2",
)

#: The two colours a "one thing against grey" plot needs. THE SAME OBJECTS
#: the saved figure uses, not a second pair chosen to look similar: a run
#: must not draw in two idioms, and a compartment that is blue on screen and
#: amber in the exported PDF is exactly that failure in miniature.
#:
#: `spacr.figures.style` is hex strings and imports no matplotlib -- measured
#: at 76 ms with matplotlib still absent from sys.modules -- so this costs a
#: GUI module nothing.
from ...figures.style import ROLES as _ROLES

HIGHLIGHT = _ROLES["highlight"]
MUTED = _ROLES["data"]

#: Points beyond this many stop getting individual hover hit-boxes, which is
#: what makes a large scatter slow to move over rather than slow to draw.
HOVER_LIMIT = 20000


def _require_pyqtgraph() -> None:
    """Raise for a caller that genuinely cannot degrade.

    NOT called from :class:`FastPlot` any more, and the reason is the bug
    this replaced. `RegressionResultsPanel` builds five of these, the
    parameter-sweep card builds that panel, and `_build_runtime_panel` builds
    the card -- so a missing OPTIONAL plotting library raised out of the
    screen factory and took down EVERY module in the application, mask and
    measure included. Reported from a real install that had PySide6 and not
    pyqtgraph.

    The message it raised also named a fallback that does not exist ("or use
    the matplotlib figures"). Telling a user there is another way and then
    dying is worse than dying.
    """
    if not HAVE_PYQTGRAPH:
        raise RuntimeError(PYQTGRAPH_MISSING_MESSAGE)


#: Ink to fall back to when the preference store cannot be read.
#:
#: White, because every spaCR theme but one is dark and a plot with invisible
#: axes is worse than a plot with slightly wrong ones. Only reached in a bare
#: process with no QSettings -- a headless render or a unit test.
_FALLBACK_FOREGROUND = "#ffffff"


def _figure_colors() -> tuple:
    """``(background, foreground)`` for a plot, from the figure preferences.

    The same source the matplotlib renderer uses
    (:func:`spacr.qt.preferences.get_figure_colors`), so the two cannot
    disagree about what a figure looks like and a theme switch moves both.
    """
    try:
        from ..preferences import get_figure_colors

        return get_figure_colors()
    except Exception:      # pragma: no cover - no settings store available
        return "none", _FALLBACK_FOREGROUND


def colour_for(index: int, alpha: int = 255) -> QColor:
    """Stable colour for category ``index``."""
    colour = QColor(PALETTE[index % len(PALETTE)])
    colour.setAlpha(alpha)
    return colour


def _finite(values) -> np.ndarray:
    """Coerce to float and replace anything unplottable with NaN.

    A p-value column arrives with blanks, strings and the occasional inf from
    a log of zero. Left alone, one of those silently rescales the whole axis
    and the plot looks empty.
    """
    array = np.asarray(values, dtype="float64")
    return np.where(np.isfinite(array), array, np.nan)


class FastPlot(QWidget):
    """A pyqtgraph plot with the controls every plot here wants.

    :ivar point_clicked: emitted with the position of a clicked point IN THIS
        PLOT'S OWN FRAME. It is not an index into anyone else's table; see
        :attr:`key_selected` for the link that survives sorting and filtering.
    :ivar key_selected: emitted with the identifier of a clicked point.
    :ivar keys_selected: emitted with the identifiers of EVERY row behind the
        thing that was clicked. A scatter point is one row and emits both; a
        histogram bar is a hundred rows and can only honestly emit this one.
    """

    point_clicked = Signal(int)
    key_selected = Signal(str)
    keys_selected = Signal(list)

    def __init__(self, title: str = "", x_label: str = "", y_label: str = "",
                 parent=None):
        super().__init__(parent)
        #: False when pyqtgraph is absent. The widget still constructs, still
        #: lays out, and says why it is empty rather than raising.
        self.plots_available = HAVE_PYQTGRAPH
        if not HAVE_PYQTGRAPH:
            self._build_without_pyqtgraph(title)
            return
        # BACKGROUND None IS TRANSPARENT, WHICH WAS ALREADY RIGHT. The ink was
        # not: `foreground="k"` hardcoded BLACK axes, ticks and labels, so on
        # a dark theme the plot drew black-on-transparent over a dark surface
        # and the axes were invisible. The matplotlib path has resolved this
        # correctly for a while via preferences.get_figure_colors(), which
        # returns TRANSPARENT_FIGURE_BG plus theme-correct ink and honours an
        # explicit colour the user has chosen; pyqtgraph simply never asked
        # it. Same source for both renderers, so a theme switch cannot move
        # one and not the other.
        self._background, self._foreground = _figure_colors()
        pg.setConfigOptions(antialias=True, background=None,
                            foreground=self._foreground)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.plot = pg.PlotWidget(title=title or None)
        self.plot.setLabel("bottom", x_label)
        self.plot.setLabel("left", y_label)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # A transparent pyqtgraph background is not enough on its own: the
        # QWidget it lives in still paints the theme's `bg` under the blanket
        # QWidget rule, so the plot sits on an opaque slab regardless. The
        # theme's own helper is what every other transparent surface here
        # uses -- see the hyperparam screen, which does exactly this.
        self.plot.setBackground(None)
        try:
            from ..theme import make_transparent

            make_transparent(self, self.plot, self.plot.viewport())
        except Exception:                   # pragma: no cover - theme absent
            pass
        layout.addWidget(self.plot, 1)

        controls = QHBoxLayout()
        self._log_x = QCheckBox("log x")
        self._log_y = QCheckBox("log y")
        for box in (self._log_x, self._log_y):
            box.toggled.connect(self._apply_log)
            controls.addWidget(box)
        self._legend_box = QCheckBox("legend")
        self._legend_box.setEnabled(False)
        self._legend_box.setToolTip(
            "Name the categories. Off by default: a 27-entry legend costs "
            "~40 ms of every redraw, against 3 ms for the plot itself.")
        self._legend_box.toggled.connect(self._toggle_legend)
        controls.addWidget(self._legend_box)

        self._grid = QCheckBox("grid")
        self._grid.setChecked(True)
        self._grid.toggled.connect(
            lambda on: self.plot.showGrid(x=on, y=on, alpha=0.25))
        controls.addWidget(self._grid)
        controls.addStretch(1)

        reset = QPushButton("Reset view")
        reset.clicked.connect(lambda: self.plot.autoRange())
        controls.addWidget(reset)
        export = QPushButton("Export…")
        export.clicked.connect(self.export)
        controls.addWidget(export)
        layout.addLayout(controls)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        #: The plot's own sentence, kept apart from whatever was last clicked.
        self._headline = ""
        #: Whatever was last clicked, kept apart from the plot's own sentence.
        self._note = ""

        self._labels: Sequence[str] = ()
        self._legend_colours: dict = {}
        self._items: list = []

        # THE KEY JOIN. Row-to-point highlighting is joined on the identifier
        # the row carries, never on a position -- a table sorted by effect and
        # a scatter drawn in input order are the same points in two orders,
        # and joining them by index lights up the WRONG guide silently, in
        # exactly the direction nobody questions, because something lit up.
        self._keys: Sequence[str] = ()
        self._key_rows: dict = {}
        self._row_xy: dict = {}
        self._selected_key: Optional[str] = None
        self._highlight = None

        #: ``[(label, callback, checked)]`` for the TAGM/LOPIT compartments
        #: this screen actually has. ONE at a time against grey; 27 hues is
        #: what the house style forbids and also what cost 40 ms of a 49 ms
        #: redraw.
        self._compartments = []

        #: ``[(label, callback, checked)]`` for the baselines this plot can
        #: measure its effects from. Empty unless the host offers them.
        self._baselines = []

        #: ``(callback, label)`` for an action that re-runs the analysis, or
        #: None. BORN HERE, not on first use: a filter control connected in
        #: __init__ to a handler that reads an attribute created later is the
        #: `_significance` crash, and it took the whole panel down at launch.
        self._refit = None

        # Right-click to restyle, the same gesture the matplotlib figures use.
        self.plot.setContextMenuPolicy(Qt.CustomContextMenu)
        self.plot.customContextMenuRequested.connect(self._style_menu)

    def _build_without_pyqtgraph(self, title: str) -> None:
        """A usable, honest empty box instead of a traceback.

        Every attribute the rest of this class and its callers touch is set
        HERE. A half-built widget that raises on its third method is worse
        than one that raises on its first: the traceback then names a symptom
        instead of the cause -- which is exactly how the original report read.
        """
        from PySide6.QtWidgets import QLabel

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)
        if title:
            layout.addWidget(QLabel(f"<b>{title}</b>"))
        notice = QLabel(PYQTGRAPH_MISSING_MESSAGE)
        notice.setWordWrap(True)
        notice.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(notice)
        layout.addStretch(1)

        self.plot = _Absorbs()
        self._background, self._foreground = _figure_colors()
        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        self._headline = ""
        self._note = ""
        self._labels = ()
        self._legend_colours = {}
        self._items = []
        self._keys = ()
        self._key_rows = {}
        self._row_xy = {}
        self._selected_key = None
        self._highlight = None
        self._refit = None
        self._baselines = []
        self._compartments = []
        self._frame = None
        self._grid = QCheckBox("Grid")
        self._legend_box = QCheckBox("Legend")
        self._legend_box.setEnabled(False)

    # ----------------------------------------------------------------- state

    def _reset_scene(self) -> None:
        """Empty the plot AND the bookkeeping that describes what was on it.

        ``plot.clear()`` takes the artists off the scene and leaves every
        dictionary that pointed at them behind. That is not tidiness: after a
        redraw ``_row_xy`` still holds the PREVIOUS table's coordinates, so
        :meth:`highlight_key` rings the place where a point used to be, and
        ``_highlight`` still names an item that is no longer in the scene, so
        removing it later raises. The volcano did this by hand; every plot
        here needs it, which is why it is one method.

        ``_keys`` is deliberately NOT cleared here -- a redraw of the same
        table must keep its identifiers -- but every ``set_*`` below re-sets
        them, so a NEW table cannot inherit the old ones either.
        """
        self.plot.clear()
        self._row_xy = {}
        self._highlight = None
        self._labels = ()

    # ------------------------------------------------------------- restyling

    def offer_refit(self, callback, label: str = "Re-fit with another model…"):
        """Add an action that CHANGES THE NUMBERS to the right-click menu.

        :param callback: called with no arguments when the user picks it.
        :param label: what the action says.

        Everything else on that menu changes how the figure looks and nothing
        else. This one re-runs the regression, so it is put under its own
        heading rather than in the list -- a user reaching for "Point size"
        must not be one slip away from starting a fit.

        Offered by the host rather than built in, because the plot knows
        nothing about settings, count data or where a run writes, and should
        not learn: the same widget draws a simulation and a sweep trial.
        """
        self._refit = (callback, label)

    def offer_compartments(self, options) -> None:
        """Offer "colour by localisation" as a submenu.

        :param options: ``[(label, callback, checked)]``.
        """
        self._compartments = list(options or ())

    def offer_baselines(self, options) -> None:
        """Offer "measure the effects from ..." on the right-click menu.

        :param options: ``[(label, callback, checked)]``.

        Separate from :meth:`offer_refit` because the two are different kinds
        of thing and a user must be able to tell them apart: a baseline moves
        where zero is drawn on a fit that has already happened, a re-fit
        replaces the fit.
        """
        self._baselines = list(options or ())

    def _style_menu(self, position) -> None:
        """Right-click: build the menu and show it."""
        self.build_style_menu().exec(self.plot.mapToGlobal(position))

    def build_style_menu(self):
        """The right-click menu, built from what the plot actually has on it.

        SEPARATE FROM SHOWING IT so the menu can be inspected without a modal
        event loop. `QMenu.exec` blocks until the user picks something and is
        not patchable from a test -- it is a C++ slot, and assigning over it
        leaves the real one dispatching -- so a test that reached in to read
        the entries hung the suite instead of failing it.
        """
        from PySide6.QtWidgets import QMenu

        menu = QMenu(self)
        menu.addAction("Point size…", self._ask_point_size)
        menu.addAction("Point colour…", self._ask_point_colour)
        menu.addAction("Opacity…", self._ask_opacity)
        menu.addSeparator()
        menu.addAction("Axis labels…", self._ask_labels)
        menu.addAction("Font size…", self._ask_font_size)
        menu.addSeparator()
        grid = menu.addAction("Grid")
        grid.setCheckable(True)
        grid.setChecked(self._grid.isChecked())
        grid.toggled.connect(self._grid.setChecked)
        if self._legend_box.isEnabled():
            legend = menu.addAction("Legend")
            legend.setCheckable(True)
            legend.setChecked(self._legend_box.isChecked())
            legend.toggled.connect(self._legend_box.setChecked)
        menu.addSeparator()
        menu.addAction("Reset view", self.plot.autoRange)
        menu.addAction("Export…", self.export)
        if self._baselines:
            # WHAT THE EFFECTS ARE MEASURED FROM. Under the restyling
            # entries because it moves the points, and above the re-fit
            # because it does NOT change the fit -- it changes where zero is
            # drawn on a fit that has already happened.
            menu.addSection("Measured from")
            for label, callback, checked in self._baselines:
                action = menu.addAction(label, callback)
                action.setCheckable(True)
                action.setChecked(bool(checked))
        if self._compartments:
            # A SUBMENU, because this is the one list that can be long -- and
            # it holds only what this screen actually has, so a choice that
            # would colour nothing is not offered at all.
            sub = menu.addMenu("Colour by localisation")
            for label, callback, checked in self._compartments:
                action = sub.addAction(label, callback)
                action.setCheckable(True)
                action.setChecked(bool(checked))
        if self._refit is not None:
            callback, label = self._refit
            # A SECTION, not another line in the list. Everything above
            # restyles; below here the numbers change.
            menu.addSection("Re-runs the analysis")
            menu.addAction(label, callback)
        return menu

    def _scatter_items(self):
        """Every scatter on the plot, for a restyle to reach.

        The selection marker is deliberately not one of them: it is a cursor,
        not data, and a restyle that shrank it to the point size would make
        the selection invisible.
        """
        return [i for i in self.plot.listDataItems()
                if hasattr(i, "setSize") and hasattr(i, "setBrush")
                and i is not self._highlight]

    def _ask_point_size(self) -> None:
        from PySide6.QtWidgets import QInputDialog

        value, ok = QInputDialog.getDouble(
            self, "Point size", "Size in pixels:", 8.0, 1.0, 60.0, 1)
        if ok:
            for item in self._scatter_items():
                item.setSize(value)

    def _ask_point_colour(self) -> None:
        from PySide6.QtWidgets import QColorDialog

        colour = QColorDialog.getColor(QColor(PALETTE[0]), self,
                                       "Point colour")
        if colour.isValid():
            # One brush for everything: this is the deliberate override of a
            # category colouring, and it is also the fastest path there is.
            brush = pg.mkBrush(colour)
            for item in self._scatter_items():
                item.setBrush(brush)

    def _ask_opacity(self) -> None:
        from PySide6.QtWidgets import QInputDialog

        value, ok = QInputDialog.getDouble(
            self, "Opacity", "0 is invisible, 1 is solid:", 1.0, 0.05, 1.0, 2)
        if ok:
            for item in self._scatter_items():
                item.setOpacity(value)

    def _ask_labels(self) -> None:
        from PySide6.QtWidgets import QInputDialog

        current_x = self.plot.getAxis("bottom").labelText
        current_y = self.plot.getAxis("left").labelText
        x, ok = QInputDialog.getText(self, "X axis label", "X:", text=current_x)
        if not ok:
            return
        y, ok = QInputDialog.getText(self, "Y axis label", "Y:", text=current_y)
        if not ok:
            return
        self.plot.setLabel("bottom", x)
        self.plot.setLabel("left", y)

    def _ask_font_size(self) -> None:
        from PySide6.QtWidgets import QInputDialog

        value, ok = QInputDialog.getInt(
            self, "Font size", "Points:", 10, 5, 40)
        if not ok:
            return
        for name in ("bottom", "left"):
            axis = self.plot.getAxis(name)
            axis.setStyle(tickFont=None)
            axis.setTickFont(None)
            label = axis.labelText
            axis.setLabel(label, **{"font-size": f"{value}pt"})

    def _apply_log(self) -> None:
        self.plot.setLogMode(self._log_x.isChecked(), self._log_y.isChecked())

    def _build_legend(self) -> None:
        """Add the legend. Only ever called when it is actually wanted."""
        colours = getattr(self, "_legend_colours", None)
        if not colours:
            return
        self.plot.addLegend(offset=(-10, 10), labelTextSize="8pt")
        for name, colour in colours.items():
            marker = pg.ScatterPlotItem(
                [], [], brush=pg.mkBrush(colour), pen=None, size=8)
            self.plot.plotItem.legend.addItem(marker, name)

    def _toggle_legend(self, on: bool) -> None:
        if on:
            self._build_legend()
            return
        legend = getattr(self.plot.plotItem, "legend", None)
        if legend is not None:
            self.plot.plotItem.legend = None
            try:
                self.plot.plotItem.scene().removeItem(legend)
            except Exception:  # pragma: no cover - already detached
                pass

    def set_status(self, text: str) -> None:
        """What this plot has to say about ITSELF. Survives a selection."""
        self._headline = text
        self._status.setText(text)

    def set_status_note(self, note: str) -> None:
        """Add a sentence about the CLICKED thing, keeping the headline.

        The diagnostics' status lines carry the numbers they exist for -- the
        inflation factor, the control medians, how many genes rest on one
        guide -- and overwriting those with the name of whatever was just
        clicked trades the panel's whole content for a string the user can
        already read in the table. Both fit.
        """
        headline = getattr(self, "_headline", "")
        self._note = note
        self._status.setText("   ".join(part for part in (headline, note)
                                        if part))

    def note_selection(self, key, found: bool) -> None:
        """Say a row was picked -- unless this plot already said MORE about it.

        A click travels: the dot announces its key, the table selects that
        row, and every other plot then marks it. That last step arrives AFTER
        the clicked plot has written its own answer, and the clicked plot
        knows the most -- which control group a dot is in and what its effect
        was, how many of a gene's guides agree, what the p-value is. The plots
        that merely received the key know only the key, so letting them write
        last replaces the answer with the question.

        :param found: whether this plot actually drew that row. ``False`` is a
            real answer and is said out loud: a coefficient with an unusable
            p-value is on no plot, a nuisance term is off the volcano on
            purpose, and a guide is not a point on a per-gene plot at all.
        """
        if found and str(key) in getattr(self, "_note", ""):
            return
        self.set_status_note(
            f"{key}" if found
            else f"{key} is in the table but not on this plot.")

    def add_scatter(self, x, y, *, colours=None, brush_list=None,
                    size: float = 8.0, labels: Sequence[str] = (),
                    symbol: str = "o", name: str = "",
                    rows=None) -> ScatterPlotItem:
        """Add points and wire up clicking them.

        :param colours: one QColor per point, or None for a single colour.
        :param labels: per-point text, shown on hover and on click.
        :param rows: the FRAME ROW each element of ``x``/``y`` came from.
            Default ``None`` means the arrays are already in frame order.

            THIS IS THE WHOLE OF THE Q-Q TRAP. A Q-Q plot is SORTED by
            p-value, so its nth drawn point is not its nth table row; a
            control panel is split into groups, so its nth point is not the
            nth row either. Left to assume otherwise, every one of those
            plots would carry an index that looks like a row, joins like a
            row, and names a different guide -- silently, and in the
            direction nobody questions, because something did light up.
        """
        x = _finite(x)
        y = _finite(y)
        keep = ~(np.isnan(x) | np.isnan(y))
        # Positions in the arrays as handed in: what indexes `colours` and
        # `brush_list`, which are drawn up alongside x and y.
        drawn = np.nonzero(keep)[0]
        # Indices into the ORIGINAL frame, so a click still identifies the
        # right row after unplottable points have been dropped -- and, when
        # `rows` says the arrays were reordered, after that too.
        original = drawn if rows is None else np.asarray(rows)[drawn]
        # ONE numpy->Python conversion, not a loop of them. `.tolist()` is a
        # C-level bulk convert; `[int(row) for row in original]` is 1,215
        # interpreter round trips and measurably re-slowed the volcano the
        # first time this was written that way.
        rows_drawn = original.tolist()

        brushes = None
        if brush_list is not None:
            # Already one reusable brush per point; nothing to build.
            brushes = [brush_list[i] for i in drawn]
        elif colours is not None:
            # ONE BRUSH PER DISTINCT COLOUR, REUSED -- not one per point.
            #
            # pg.mkBrush() per point builds 1,215 QBrush objects and defeats
            # pyqtgraph's fast path completely. Measured on the real volcano:
            #
            #     a brush constructed per point      39.5 ms
            #     27 brushes, indexed per point       3.5 ms
            #     a single brush for everything       1.6 ms
            #
            # The colours themselves were never the problem; allocating them
            # was. This is the whole of the lag on the last graph.
            colours = list(colours)
            cache: dict = {}
            brushes = []
            for i in drawn:
                colour = colours[i]
                key = colour.rgba() if hasattr(colour, "rgba") else str(colour)
                brush = cache.get(key)
                if brush is None:
                    brush = cache[key] = pg.mkBrush(colour)
                brushes.append(brush)

        # `data` must go in with the points: calling setData afterwards ADDS
        # points rather than annotating the ones already there.
        item = pg.ScatterPlotItem(
            x=x[keep], y=y[keep], size=size, symbol=symbol,
            pen=pg.mkPen(None),
            brush=brushes if brushes is not None else pg.mkBrush(colour_for(0)),
            hoverable=len(drawn) <= HOVER_LIMIT,
            data=rows_drawn, name=name or None,
        )
        item.sigClicked.connect(self._on_points_clicked)
        self.plot.addItem(item)
        # Where each row ended up, so a selection arriving later can be drawn
        # without re-deriving the transform that put it there.
        self._row_xy.update(zip(rows_drawn,
                                zip(x[keep].tolist(), y[keep].tolist())))
        if labels is not None and len(labels):
            self._labels = labels
        return item

    # ------------------------------------------------------------ selection

    def set_keys(self, keys) -> int:
        """Give each frame row its identifier. Returns the number of rows.

        Duplicates are kept as the FIRST row carrying the key, and counted:
        an identifier that names two rows cannot select one of them, and
        picking silently is how the wrong point gets highlighted.

        ``None`` -- either as the whole argument or as one entry -- means
        "this row has no identifier". Such a row still draws and can still be
        clicked; it simply reports nothing to anyone else, which is the
        truthful answer and is not the same as reporting the empty string,
        which would collide with every other unidentified row.
        """
        if keys is None:
            keys = ()
        # A MISSING KEY IS None, NOT THE STRING "nan". A frame column carries
        # its blanks as float NaN, and str() turns every one of them into the
        # same four characters -- which would make one bogus identifier that
        # several unrelated rows answer to, i.e. exactly the collision this
        # method's duplicate rule exists to prevent.
        self._keys = [None if key is None or key != key else str(key)
                      for key in keys]
        self._key_rows = {}
        for row, key in enumerate(self._keys):
            if key is not None:
                self._key_rows.setdefault(key, row)
        return len(self._keys)

    def _has_usable_keys(self) -> bool:
        """Whether ANY row on this plot can be identified to anyone else.

        NOT ``bool(self._keys)``. A caller can hand over a full-length column
        of blanks -- a fit with no gene-level terms gives the agreement plot
        exactly that, one ``None`` per gene -- and a list of ``None`` is
        truthy. Read that way, the plot appends "Click a point for its
        coefficient" to its status line while every click resolves to no key
        and selects nothing, which is an invitation it cannot honour.
        """
        return any(key is not None for key in self._keys)

    def key_for_row(self, row: int) -> Optional[str]:
        """The identifier at frame position ``row``, if this plot has keys."""
        if self._keys and 0 <= int(row) < len(self._keys):
            return self._keys[int(row)]
        return None

    def highlight_key(self, key) -> bool:
        """Ring the point identified by ``key``. Returns whether one was found.

        ``False`` is a real answer, not a failure: a key can be absent because
        its point was not plotted (an unusable p-value) or because it is a
        nuisance term this plot deliberately leaves off. Saying so beats
        ringing something near it.
        """
        key = None if key is None else str(key)
        self._selected_key = key
        if self._highlight is not None:
            try:
                self.plot.removeItem(self._highlight)
            except Exception:               # pragma: no cover - already gone
                pass
            self._highlight = None
        if key is None:
            return False
        row = self._key_rows.get(key)
        if row is None:
            return False
        return self._draw_marker(row)

    def _draw_marker(self, row: int) -> bool:
        """Mark the row at frame position ``row``. False if it is not drawn.

        Split out so a plot whose marks are not points can say so its own way
        -- a histogram bar cannot be ringed like a dot -- while the key
        lookup, the clearing and the "was it found" answer stay in one place.
        """
        position = self._row_xy.get(row)
        if position is None:
            return False
        x, y = position
        # An open ring, not a filled dot: filling it would hide the point it
        # is meant to identify, including its category colour.
        self._highlight = pg.ScatterPlotItem(
            x=[x], y=[y], symbol="o", size=20, brush=pg.mkBrush(None),
            pen=pg.mkPen(QColor(self._foreground), width=2.0))
        self._highlight.setZValue(50)
        self.plot.addItem(self._highlight)
        return True

    def clear_highlight(self) -> None:
        self.highlight_key(None)

    def add_line(self, *, x=None, y=None, colour: str = "#C44E52",
                 style=Qt.DashLine, width: float = 1.5, label: str = ""):
        """A threshold line. ``x`` for vertical, ``y`` for horizontal."""
        pen = pg.mkPen(QColor(colour), width=width, style=style)
        line = pg.InfiniteLine(
            pos=(x if x is not None else y),
            angle=90 if x is not None else 0,
            pen=pen, label=label or None,
            labelOpts={"position": 0.92, "color": colour, "movable": False},
        )
        self.plot.addItem(line)
        return line

    def _on_points_clicked(self, _item, points) -> None:
        if not len(points):
            return
        index = points[0].data()
        if index is None:
            return
        index = int(index)
        text = "   ".join(part for part in (self._describe(index),
                                            self._detail(index)) if part)
        if text:
            self.set_status_note(text)
        key = self.key_for_row(index)
        if key is not None:
            self.highlight_key(key)
            self.key_selected.emit(key)
            self.keys_selected.emit([key])
        self.point_clicked.emit(index)

    def _describe(self, index: int) -> str:
        """Describe ONE point, on demand.

        Formatting every point up front is what made the plot slow to appear;
        formatting the clicked one costs nothing and reads the same.
        """
        if self._labels is not None and index < len(self._labels or ()):
            return str(self._labels[index])
        frame = getattr(self, "_frame", None)
        if frame is not None and index < len(frame):
            parts = []
            for column in (getattr(self, "_label_column", None),
                           getattr(self, "_effect_column", None),
                           getattr(self, "_p_column", None)):
                if column and column in frame.columns:
                    value = frame[column].iloc[index]
                    parts.append(f"{column}={value}"
                                 if not isinstance(value, str) else str(value))
            if parts:
                return "   ".join(parts)
        # THE IDENTIFIER IS ALREADY THE ANSWER. A diagnostic plot holds no
        # frame -- it is handed an array of p-values -- so without this a
        # click on the Q-Q reported an empty status line while quietly
        # selecting the right row somewhere else. The key IS the guide's
        # name; saying it costs one lookup and is what the user clicked for.
        key = self.key_for_row(index)
        return key or ""

    def _detail(self, index: int) -> str:
        """Whatever THIS plot knows about the row that the key does not.

        A hook, not a table read: the point of formatting on click is that no
        per-point work happens before one. Subclasses that already hold the
        plotted arrays answer from them in O(1).
        """
        return ""

    # ---------------------------------------------------------------- export

    def export(self, path: Optional[str] = None) -> Optional[str]:
        """Write the plot out. SVG when the name says so, else PNG.

        pyqtgraph's SVG export is real vector output, so what is exported from
        the screen is publishable rather than a screenshot of it.
        """
        if path is None:
            from PySide6.QtWidgets import QFileDialog
            path, _ = QFileDialog.getSaveFileName(
                self, "Export plot", "plot.png",
                "Vector (*.svg);;Image (*.png)")
            if not path:
                return None
        from pyqtgraph import exporters

        item = self.plot.plotItem
        if str(path).lower().endswith(".svg"):
            exporters.SVGExporter(item).export(path)
        else:
            exporter = exporters.ImageExporter(item)
            # MATCH THE SCREEN. The exporter defaults to pyqtgraph's config
            # background, so a plot drawn transparent was saved onto an opaque
            # slab -- which is the one thing the maintainer asked for by name
            # ("not black not white just transparent") and the one place a
            # transparent plot would have quietly stopped being transparent.
            try:
                exporter.parameters()["background"] = QColor(0, 0, 0, 0)
            except (KeyError, TypeError):   # pragma: no cover - older pyqtgraph
                pass
            exporter.export(path)
        return path

    def restyle(self, background: Optional[str] = None,
                foreground: Optional[str] = None) -> None:
        """Re-read the figure colours, or take the ones given.

        Needed because pyqtgraph resolves ``foreground`` at construction:
        without this a theme switch leaves every open plot drawing its old
        ink, and on a dark-to-light switch that ink is invisible.
        """
        if background is None or foreground is None:
            resolved_bg, resolved_fg = _figure_colors()
            background = resolved_bg if background is None else background
            foreground = resolved_fg if foreground is None else foreground
        self._background, self._foreground = background, foreground
        pg.setConfigOptions(foreground=foreground)
        axis_pen = pg.mkPen(foreground)
        for edge in ("bottom", "left", "top", "right"):
            try:
                axis = self.plot.getAxis(edge)
            except Exception:               # pragma: no cover - absent axis
                continue
            axis.setPen(axis_pen)
            axis.setTextPen(axis_pen)
        title = getattr(self.plot.plotItem, "titleLabel", None)
        if title is not None and title.text:
            self.plot.setTitle(title.text, color=foreground)


class VolcanoPlot(FastPlot):
    """Effect against -log10(p), coloured by a category, every dot clickable."""

    def __init__(self, parent=None):
        super().__init__(title="Volcano", x_label="coefficient",
                         y_label="-log10(p)", parent=parent)

    def set_results(self, frame, *, effect: str = "coefficient",
                    p_column: str = "p_value", label_column: str = "feature",
                    category_column: Optional[str] = None,
                    alpha: float = 0.05,
                    effect_threshold: Optional[float] = None,
                    key_column: Optional[str] = None,
                    drop_untested: bool = True,
                    compartment: Optional[str] = None):
        """Draw ``frame``. Returns the number of points actually plotted.

        :param compartment: one TAGM/LOPIT compartment to pick out against
            grey. ONE, not all 27 -- see :mod:`spacr.localisation`. It
            REPLACES any category colouring rather than combining with it: a
            volcano where a coloured dot might be coloured for its condition
            or for its compartment has no sentence.
        """
        self._reset_scene()
        if frame is None or not len(frame):
            self.set_status("No coefficients to plot.")
            return 0

        # NUISANCE TERMS ARE NOT HYPOTHESES, AND THEY OWN THE AXIS.
        #
        # The intercept and the plate row/column effects are covariates: they
        # are fitted so the guide effects come out clean, not so anyone can
        # ask whether they differ from zero. spacr.ml already draws that line
        # -- it leaves them out of the multiple-testing family, which is why
        # they leave a fit with q_value = NaN -- and plotting them draws a
        # different experiment from the one the q-values describe.
        #
        # It is not a rounding error. On plate1_dv the intercept sits at
        # -log10(p) = 45.5 against 12.5 for the strongest real hit and 2.3 at
        # the 99th percentile, so ONE untestable row makes the y-axis 3.6x
        # taller than the data and flattens the whole screen into the bottom
        # of it. A fit carrying row and column terms has ~25 of them.
        untested = 0
        if drop_untested and "feature" in getattr(frame, "columns", ()):
            from ...hits import tested_family

            keep_rows = tested_family(frame["feature"])
            if not keep_rows.all():
                untested = int((~keep_rows).sum())
                frame = frame.loc[keep_rows].reset_index(drop=True)
                if not len(frame):
                    self.set_status(
                        f"No testable coefficients: all {untested} rows are "
                        "nuisance terms.")
                    return 0

        effects = _finite(frame[effect]) if effect in frame else np.zeros(len(frame))
        p_values = _finite(frame[p_column]) if p_column in frame \
            else np.full(len(frame), np.nan)
        # A p of exactly zero is a real result underflowing, not a mistake;
        # clamping keeps it on the plot instead of sending it to infinity.
        smallest = np.nanmin(p_values[p_values > 0]) if np.any(p_values > 0) \
            else 1e-300
        neglog = -np.log10(np.clip(p_values, smallest * 1e-3, 1.0))

        brush_list, legend = None, {}
        if compartment:
            # ONE COMPARTMENT AGAINST GREY. Two brushes and a two-entry
            # legend: the 27-colour version is what the house style forbids
            # and, measured, its legend cost 40 ms of a 49 ms redraw.
            from ...localisation import mask as compartment_mask

            inside = compartment_mask(frame, compartment).to_numpy()
            if inside.any():
                here = pg.mkBrush(HIGHLIGHT)
                elsewhere = pg.mkBrush(MUTED)
                brush_list = [here if flag else elsewhere for flag in inside]
                legend = {f"{compartment} ({int(inside.sum())})": HIGHLIGHT,
                          f"{int((~inside).sum())} elsewhere": MUTED}
        elif category_column and category_column in frame:
            # Categorical codes are computed in C; the alternative is a Python
            # loop over 1,215 pandas values plus a QColor.rgba() per point,
            # which cost 45 ms of the 48 ms this used to take.
            import pandas as _pd

            categorical = _pd.Categorical(frame[category_column].astype(str))
            names = list(categorical.categories)
            legend = {name: colour_for(i) for i, name in enumerate(names)}
            palette = [pg.mkBrush(legend[name]) for name in names]
            unknown = pg.mkBrush(colour_for(0))
            brush_list = [palette[c] if c >= 0 else unknown
                          for c in categorical.codes]

        # NO PER-POINT WORK BEFORE DRAWING.
        #
        # This used to build a label string for all 1,215 rows up front, three
        # `frame[col].iloc[i]` lookups each. Pandas scalar indexing in a Python
        # loop is ~3,600 lookups to draw a scatter plot, and it cost more than
        # the drawing did. The frame is kept instead and a label is formatted
        # for the ONE point that gets clicked -- which is the only one anybody
        # ever reads.
        self._frame = frame
        self._label_column = label_column
        self._effect_column = effect
        self._p_column = p_column
        self._labels = ()

        # `feature` is the design-matrix term name and is one-to-one with the
        # row -- checked on the real screen: 1,213 rows, 1,213 distinct. `gene`
        # and `grna` are NOT keys, because a gene has several guides and
        # several rows, so joining on either highlights an arbitrary one.
        key = key_column or ("feature" if "feature" in frame.columns
                             else label_column)
        self.set_keys(frame[key] if key in frame.columns else frame.index)

        self.add_scatter(effects, neglog, brush_list=brush_list)
        self.add_line(y=-np.log10(alpha), label=f"p={alpha:g}")
        if effect_threshold:
            for sign in (-1, 1):
                self.add_line(x=sign * abs(effect_threshold), colour="#8C8C8C")

        # THE LEGEND IS OPT-IN, AND IT IS THE REASON WHY.
        #
        # Twenty-seven entries cost 40 ms of a 49 ms redraw -- each one builds
        # a ScatterPlotItem and a LabelItem. It is the identical cost that made
        # matplotlib's version 63 ms, so bringing it across unchanged would
        # have carried the lag over to the new library and wasted the switch.
        #
        #     scatter alone, 1,215 points        3.4 ms
        #     the same plus a 27-entry legend   43.7 ms
        #
        # So the plot draws without one and offers a checkbox. Colour still
        # identifies the compartments; the legend only names them, and naming
        # them is worth 40 ms when asked for and not before.
        self._legend_colours = legend
        if legend:
            self._legend_box.setEnabled(True)
            self._legend_box.setText(f"legend ({len(legend)})")
            if self._legend_box.isChecked():
                self._build_legend()
        else:
            self._legend_box.setEnabled(False)

        # A SELECTION SURVIVES A REDRAW. plot.clear() took the marker with it,
        # so it goes back on -- otherwise changing the colouring, or any other
        # setting, silently deselects whatever the user was looking at.
        if self._selected_key is not None:
            self.highlight_key(self._selected_key)

        plotted = int(np.sum(~(np.isnan(effects) | np.isnan(neglog))))
        note = f"{plotted} coefficients."
        if untested:
            # Reported, not silently removed: the difference between a filter
            # and a lie.
            note += (f" {untested} nuisance "
                     f"term{'s' if untested != 1 else ''} not shown (not "
                     "tested, so no q-value).")
        self.set_status(f"{note} Click a point for detail.")
        return plotted


class PValueHistogram(FastPlot):
    """The single most informative check that a correction means anything.

    Under the null, p-values are uniform. A histogram that is flat with a spike
    at zero is a screen with real hits in it; one that slopes, or piles up near
    one, says the model is misspecified and every q-value downstream of it is
    decoration.
    """

    def __init__(self, parent=None):
        super().__init__(title="p-value distribution", x_label="p",
                         y_label="count", parent=parent)
        self._edges: Optional[np.ndarray] = None
        self._counts: Optional[np.ndarray] = None
        self._bin_rows: list = []
        self._row_bin: np.ndarray = np.empty(0, dtype="int64")
        self._p: np.ndarray = np.empty(0)
        # A BAR IS NOT A POINT, so there is no sigClicked to connect to. The
        # scene reports where the user pressed and the bin is worked out from
        # the x coordinate, which is also the only definition of "which bar"
        # that stays right when the axis is zoomed.
        self.plot.scene().sigMouseClicked.connect(self._on_scene_clicked)

    def set_p_values(self, values, bins: int = 50, *, keys=None):
        """Draw the histogram. Returns the number of usable p-values.

        :param keys: one identifier per element of ``values``, in frame order.
            Given them, clicking a bar names the coefficients inside it.
        """
        self._reset_scene()
        self.set_keys(keys)
        self._edges = self._counts = None
        self._bin_rows = []
        self._row_bin = np.empty(0, dtype="int64")
        p = _finite(values)
        self._p = p
        rows = np.nonzero(~np.isnan(p))[0]
        if not len(rows):
            self.set_status("No p-values.")
            return 0
        held = p[rows]
        counts, edges = np.histogram(held, bins=bins, range=(0.0, 1.0))
        bars = pg.BarGraphItem(x0=edges[:-1], x1=edges[1:], height=counts,
                               brush=pg.mkBrush(colour_for(0, 190)),
                               pen=pg.mkPen(None))
        self.plot.addItem(bars)
        expected = len(held) / bins
        self.add_line(y=expected, colour="#C44E52", label="uniform")

        # WHICH ROWS ARE IN WHICH BAR, worked out the same way np.histogram
        # decided the counts -- half-open bins with the last one closed at 1
        # -- so the number a bar draws and the number of rows it hands back
        # are the same number. A p outside [0, 1] is in no bar at all, which
        # is exactly what np.histogram's `range` did with it.
        # Vectorised, because a bar chart must not pay a per-COEFFICIENT
        # Python loop to be drawn -- that is the cost this whole module exists
        # to avoid, and a screen has as many p-values as it has coefficients.
        inside = (held >= edges[0]) & (held <= edges[-1])
        placed = np.clip(np.searchsorted(edges, held, side="right") - 1,
                         0, bins - 1)
        self._edges, self._counts = edges, counts
        members, in_bin = rows[inside], placed[inside]
        order = np.argsort(in_bin, kind="stable")
        cuts = np.cumsum(np.bincount(in_bin, minlength=bins))[:-1]
        self._bin_rows = list(np.split(members[order], cuts))
        # Row -> its bar, as a dense array rather than a dict of thousands.
        self._row_bin = np.full(len(p), -1, dtype="int64")
        self._row_bin[members] = in_bin

        excess = max(int(counts[0] - expected), 0)
        note = (f"{len(held)} p-values. The flat line is what a screen with "
                f"no signal would give; the first bin holds {excess} more "
                f"than that.")
        if self._has_usable_keys():
            note += " Click a bar for what is in it."
        self.set_status(note)
        if self._selected_key is not None:
            self.highlight_key(self._selected_key)
        return len(held)

    # ------------------------------------------------------------- clicking

    def bin_at(self, x) -> Optional[int]:
        """The bar under data coordinate ``x``, or ``None`` beyond the axis."""
        if self._edges is None or not len(self._bin_rows):
            return None
        if x < self._edges[0] or x > self._edges[-1]:
            return None
        index = int(np.searchsorted(self._edges, x, side="right") - 1)
        return int(np.clip(index, 0, len(self._bin_rows) - 1))

    def keys_in_bin(self, index: int) -> list:
        """Every identifier the bar at ``index`` was built from."""
        if not 0 <= int(index) < len(self._bin_rows):
            return []
        found = (self.key_for_row(int(row)) for row in self._bin_rows[index])
        return [key for key in found if key is not None]

    def select_bin(self, index: int) -> list:
        """Answer a click on one bar. Returns the identifiers inside it.

        A BAR HOLDING A HUNDRED COEFFICIENTS CANNOT SELECT ONE OF THEM, and
        picking the first, the strongest or the nearest would be a guess
        dressed up as an answer -- the same mistake as joining on a position.
        So the honest split: a bar that holds exactly one row selects it like
        any other point, and a bar that holds more says what it holds and
        hands the whole set over for the table to narrow to.
        """
        keys = self.keys_in_bin(index)
        if self._edges is None or not 0 <= int(index) < len(self._bin_rows):
            return []
        low, high = float(self._edges[index]), float(self._edges[index + 1])
        count = len(self._bin_rows[index])
        span = f"p {low:.3g} to {high:.3g}"
        if not count:
            self.set_status_note(f"{span}: empty.")
            return []
        if len(keys) == 1:
            self.highlight_key(keys[0])
            self.set_status_note(f"{span}: {keys[0]}")
            self.key_selected.emit(keys[0])
        else:
            self.highlight_bin(index)
            named = f", {len(keys)} of them named" if 0 < len(keys) < count \
                else ""
            self.set_status_note(
                f"{span}: {count} coefficient{'s' if count != 1 else ''}"
                f"{named}. A bar is not one point, so this selects the set "
                f"rather than guessing which of them you meant.")
        if keys:
            self.keys_selected.emit(list(keys))
        return keys

    def _on_scene_clicked(self, event) -> None:
        """A press anywhere on the plot, mapped to the bar under it."""
        try:
            if event.button() != Qt.LeftButton:
                return
            position = event.scenePos()
            item = self.plot.plotItem
            if not item.sceneBoundingRect().contains(position):
                return
            point = item.vb.mapSceneToView(position)
        except Exception:          # pragma: no cover - no viewbox to map into
            return
        index = self.bin_at(float(point.x()))
        if index is not None:
            self.select_bin(index)

    def highlight_bin(self, index: int) -> bool:
        """Outline one bar. The histogram's answer to ringing a point."""
        if self._edges is None or self._counts is None:
            return False
        if not 0 <= int(index) < len(self._counts):
            return False
        if self._highlight is not None:
            try:
                self.plot.removeItem(self._highlight)
            except Exception:           # pragma: no cover - already gone
                pass
        # An OUTLINE, not a refill: the same reason the scatter marker is an
        # open ring. A solid bar in the highlight colour would hide how tall
        # it is against its neighbours, which is the only thing the panel is
        # for.
        self._highlight = pg.BarGraphItem(
            x0=[self._edges[index]], x1=[self._edges[index + 1]],
            height=[self._counts[index]], brush=pg.mkBrush(None),
            pen=pg.mkPen(QColor(self._foreground), width=2.0))
        self._highlight.setZValue(50)
        self.plot.addItem(self._highlight)
        return True

    def _draw_marker(self, row: int) -> bool:
        """A row selected elsewhere marks THE BAR IT FALLS IN.

        Not a ring floating over the bars: this plot never drew that row as a
        mark of its own, and inventing one would put a point on a histogram
        where there is no point. The bar is where the coefficient actually is.
        """
        row = int(row)
        if not 0 <= row < len(self._row_bin):
            return False
        index = int(self._row_bin[row])
        if index < 0:
            return False
        return self.highlight_bin(index)

    def _detail(self, index: int) -> str:
        if index < len(self._p) and np.isfinite(self._p[index]):
            return f"p = {self._p[index]:.3g}"
        return ""


class QQPlot(FastPlot):
    """Observed against expected quantiles -- is the null calibrated?

    Points on the diagonal mean the test is behaving. A curve that lifts off it
    early means inflation: the design is confounded, and the hits at the top of
    the volcano are partly an artefact of that rather than biology.
    """

    def __init__(self, parent=None):
        super().__init__(title="p-value Q-Q", x_label="expected -log10(p)",
                         y_label="observed -log10(p)", parent=parent)
        self._p: np.ndarray = np.empty(0)

    def set_p_values(self, values, *, keys=None):
        """Draw the Q-Q. Returns the number of usable tests.

        :param keys: one identifier per element of ``values``, IN THE ORDER
            THEY WERE HANDED IN -- i.e. in frame order, including the ones
            with no usable p-value. Given them, every point is clickable and
            selects the coefficient it was computed from.

        THE SORT IS THE TRAP. A Q-Q is ranked by p, so the nth drawn point is
        the nth SMALLEST p-value and almost never the nth row of the table.
        The rows are therefore carried through the sort explicitly (``rows=``)
        rather than re-derived from the drawing order, which is the mistake
        that lights up the wrong guide and looks entirely correct doing it.
        """
        self._reset_scene()
        self.set_keys(keys)
        p = _finite(values)
        self._p = p
        # Frame rows, kept alongside their p-values through the sort.
        rows = np.nonzero(~np.isnan(p) & (p > 0))[0]
        if not len(rows):
            self.set_status("No usable p-values.")
            return 0
        rows = rows[np.argsort(p[rows], kind="stable")]
        n = len(rows)
        expected = -np.log10((np.arange(1, n + 1) - 0.5) / n)
        observed = -np.log10(p[rows])
        self.add_scatter(expected, observed, size=6, rows=rows)
        top = float(max(expected.max(), observed.max()))
        self.plot.plot([0, top], [0, top],
                       pen=pg.mkPen("#C44E52", width=1.5, style=Qt.DashLine))
        # Genomic inflation: the ratio at the median. 1.0 is calibrated.
        chi = np.median(observed) / np.median(expected) if np.median(expected) else float("nan")
        note = (f"{n} tests. Inflation at the median is {chi:.2f} "
                f"(1.00 is calibrated; well above it means the null is not "
                f"flat).")
        if self._has_usable_keys():
            note += " Click a point for its coefficient."
        self.set_status(note)
        # A SELECTION SURVIVES A REDRAW, here for the same reason it does on
        # the volcano: the user picked a guide, and reloading or recolouring
        # is not them un-picking it.
        if self._selected_key is not None:
            self.highlight_key(self._selected_key)
        return n

    def _detail(self, index: int) -> str:
        if index < len(self._p) and np.isfinite(self._p[index]):
            return f"p = {self._p[index]:.3g}"
        return ""


class ResidualPlot(FastPlot):
    """Residual against fitted -- the check for a mis-specified mean.

    A horizontal band is what a well-specified model gives. A funnel means the
    variance grows with the fit and the standard errors are wrong, which is a
    p-value problem rather than a cosmetic one.
    """

    def __init__(self, parent=None):
        super().__init__(title="Residuals vs fitted", x_label="fitted",
                         y_label="residual", parent=parent)

    def set_residuals(self, fitted, residuals, labels: Sequence[str] = ()):
        self._reset_scene()
        f, r = _finite(fitted), _finite(residuals)
        if not len(f):
            self.set_status("No residuals.")
            return 0
        self.add_scatter(f, r, size=6, labels=labels)
        self.add_line(y=0.0, colour="#C44E52")
        good = ~(np.isnan(f) | np.isnan(r))
        if good.sum() > 2:
            # A crude trend line: if this is not flat, the mean is wrong.
            slope, intercept = np.polyfit(f[good], r[good], 1)
            xs = np.array([np.nanmin(f), np.nanmax(f)])
            self.plot.plot(xs, slope * xs + intercept,
                           pen=pg.mkPen("#DD8452", width=1.5))
            self.set_status(
                f"{int(good.sum())} residuals. Trend slope {slope:+.3g} -- "
                f"far from zero means the mean model is missing something.")
        return int(good.sum())


class ControlSeparation(FastPlot):
    """How far apart the positive and negative controls sit.

    This is the assay window. If the controls do not separate, nothing further
    down the pipeline can be trusted, and it is worth seeing before the volcano
    rather than after arguing about a hit list.
    """

    def __init__(self, parent=None):
        super().__init__(title="Control separation", x_label="",
                         y_label="effect", parent=parent)
        self._effects: np.ndarray = np.empty(0)
        #: ``(start, stop, group name)`` per group over the flat row space.
        self._spans: list = []

    def set_groups(self, groups: dict, *, keys: Optional[dict] = None):
        """Draw the groups. Returns the number of points plotted.

        :param groups: ``{'negative': array, 'positive': array, ...}``
        :param keys: ``{'negative': identifiers, ...}``, one identifier per
            value of the SAME group, in the same order. Given them, every dot
            is clickable and selects the coefficient behind it.

        THE GROUPS ARE THE SECOND FORM OF THE SORT TRAP. These arrays are
        slices of the table taken by condition, so a dot's position within
        its own group is not its row -- and the negative controls are drawn
        before the screen, so it is not its position on the plot either. Rows
        are therefore laid out in one flat sequence up front and carried into
        each scatter, rather than being inferred from the drawing order.
        """
        self._reset_scene()
        self._spans = []
        if not groups:
            self.set_keys(())
            self._effects = np.empty(0)
            self.set_status("No controls identified.")
            return 0

        # One flat row space over every group, so a key means the same thing
        # whichever group it came from.
        flat_keys: list = []
        columns: list = []
        base: dict = {}
        for name, values in groups.items():
            v = _finite(values)
            base[name] = len(flat_keys)
            given = None if keys is None else keys.get(name)
            if given is None:
                flat_keys.extend([None] * len(v))
            else:
                given = list(given)
                # A short or long key list is a caller bug that would silently
                # shift every row after it; pad rather than mis-join.
                if len(given) != len(v):
                    given = [given[i] if i < len(given) else None
                             for i in range(len(v))]
                flat_keys.extend(given)
            columns.append(v)
            # WHICH GROUP A ROW IS IN, AS A SPAN RATHER THAN AS 1,186 DICT
            # ENTRIES. Only the clicked point is ever asked, and the module's
            # whole performance argument is that nothing is computed per point
            # before a click. Three tuples answer it in a scan of three.
            self._spans.append((base[name], base[name] + len(v), name))
        self.set_keys(flat_keys if keys is not None else ())
        self._effects = np.concatenate(columns) if columns else np.empty(0)

        summary, total = [], 0
        rng = np.random.default_rng(0)
        for position, (name, values) in enumerate(groups.items()):
            v = _finite(values)
            finite = np.nonzero(~np.isnan(v))[0]
            if not len(finite):
                continue
            total += len(finite)
            # Jitter, so overlapping points stay countable by eye.
            x = position + (rng.random(len(finite)) - 0.5) * 0.35
            self.add_scatter(x, v[finite], size=7, rows=base[name] + finite,
                             colours=[colour_for(position, 200)] * len(finite))
            median = float(np.median(v[finite]))
            # THE MEDIAN IS THE SENTENCE OF THIS PANEL -- whether the classes
            # separate is read off these two lines -- so it is drawn in the
            # plot's ink. It was hardcoded BLACK, which on every dark spaCR
            # theme but one is a line nobody can see, and it is the one mark
            # here that must be visible.
            self.plot.plot([position - 0.25, position + 0.25], [median, median],
                           pen=pg.mkPen(QColor(self._foreground), width=2))
            summary.append(f"{name} n={len(finite)} median={median:.3g}")
        axis = self.plot.getAxis("bottom")
        axis.setTicks([[(i, name) for i, name in enumerate(groups)]])
        note = "   ".join(summary) if summary else "No control values."
        if self._has_usable_keys() and summary:
            note += "   Click a point for its coefficient."
        self.set_status(note)
        if self._selected_key is not None:
            self.highlight_key(self._selected_key)
        return total

    def group_of(self, row: int) -> Optional[str]:
        """Which group the flat row ``row`` belongs to."""
        for start, stop, name in self._spans:
            if start <= int(row) < stop:
                return name
        return None

    def _detail(self, index: int) -> str:
        parts = []
        name = self.group_of(index)
        if name:
            parts.append(str(name))
        if index < len(self._effects) and np.isfinite(self._effects[index]):
            parts.append(f"effect = {self._effects[index]:.3g}")
        return "   ".join(parts)


class GuideAgreementPlot(FastPlot):
    """Per gene: do its own guides push the same way?

    The interactive twin of :func:`spacr.figures.panels.guide_agreement`, and
    the one thing a volcano structurally cannot show. A gene called by one
    guide out of six and a gene whose six guides agree are the same dot on a
    volcano, ranked by the same number, and only one of them is corroborated
    evidence.

    Measured on the TSG101 screen: 389 genes, of which 102 rest on a single
    surviving guide -- including 244480, whose gene-level p of 2.9e-13 ranks
    it above everything else in the screen and IS that one guide's p-value.

    THE HOUSE RULE DECIDES THE COLOURING. Everything is grey except what the
    sentence is about, and the sentence here is "these ones rest on a single
    guide", so those are the only points that get colour.
    """

    #: Default ink for a gene whose guides corroborate each other.
    GREY = "#B4B4B4"
    #: The argument: a gene with nothing to corroborate it.
    SINGLE = "#C44E52"

    def __init__(self, parent=None):
        super().__init__(title="Guide agreement", x_label="guides per gene",
                         y_label="fraction agreeing in sign", parent=parent)
        self._support = None
        self._rows_shown: np.ndarray = np.empty(0, dtype="int64")

    def set_support(self, support, *, keys=None, key_column: str = "feature"):
        """Draw one point per gene. Returns the number of genes plotted.

        :param support: the frame :func:`spacr.guide_concordance.guide_support`
            returns -- ``n_guides``, ``concordance``, ``single_guide`` per
            gene -- indexed by gene or carrying a ``gene`` column.
        :param keys: one identifier per row of ``support``. Default is
            ``support[key_column]`` when that column is there.

        THE KEY IS THE GENE-LEVEL TERM, NOT THE GENE ID. A gene appears in
        the coefficient table as ``gene_fraction:gene[244480]``, and that is
        what the volcano, the table and the gene tile all join on. Handing
        this plot the bare ``244480`` would make a second key space that
        nothing else can resolve, so the caller passes the term and clicking
        a gene here selects exactly the row clicking its dot on the volcano
        would.
        """
        self._reset_scene()
        self._support = support
        self._rows_shown = np.empty(0, dtype="int64")
        if support is None or not len(support):
            self.set_keys(())
            self.set_status("No guide-level terms were fitted, so guide "
                            "support is unknown.")
            return 0

        frame = support.reset_index() if support.index.name else support
        if keys is None and key_column in getattr(frame, "columns", ()):
            keys = frame[key_column]
        self.set_keys(keys)

        counts = _finite(frame["n_guides"]) if "n_guides" in frame \
            else np.full(len(frame), np.nan)
        agree = _finite(frame["concordance"]) if "concordance" in frame \
            else np.full(len(frame), np.nan)
        single = (np.asarray(frame["single_guide"], dtype=bool)
                  if "single_guide" in frame else counts <= 1)

        # JITTERED, for the same reason the static panel is: guides per gene
        # is a small integer and agreement is a handful of fractions, so
        # several hundred genes stack into a dozen dots and the panel looks
        # like it holds no data. Seeded, so the picture is the same every
        # time, and recorded per row -- the ring a selection draws reads its
        # coordinates back out of `_row_xy`, so it lands on the dot the user
        # actually sees rather than on the un-jittered lattice point.
        rng = np.random.default_rng(0)
        x = counts + rng.uniform(-0.22, 0.22, len(frame))

        rows = np.arange(len(frame))
        for mask, colour, size in ((~single, self.GREY, 7),
                                   (single, self.SINGLE, 9)):
            if not mask.any():
                continue
            picked = rows[mask]
            self.add_scatter(x[picked], agree[picked], size=size,
                             rows=picked,
                             colours=[QColor(colour)] * len(picked))
        self._rows_shown = rows

        self.add_line(y=0.5, colour=self.GREY, label="chance")

        # ONE GENE MUST NOT OWN THE AXIS. The library gives a gene two to four
        # guides; the non-targeting control block parses as a single "gene"
        # carrying all 24 of them, and on autorange that ONE point stretches
        # the x-axis six times wider than the data and squashes all 388 real
        # genes into the left fifth of the panel. It is the identical failure
        # the intercept caused on the volcano, measured the same way.
        #
        # The volcano's answer was to DROP the offender, because a nuisance
        # term is not a hypothesis. This one is different: an over-represented
        # gene is still a gene, and dropping it would lose a real point. So it
        # is drawn and merely left outside the OPENING view -- "Reset view"
        # reaches it, and the status says it is out there rather than letting
        # it disappear silently.
        beyond = 0
        finite = counts[np.isfinite(counts)]
        if len(finite):
            bound = max(4.0, float(np.ceil(np.percentile(finite, 99))))
            beyond = int(np.sum(finite > bound))
            self.plot.setXRange(0.5, bound + 0.5, padding=0.02)

        drawn = int(np.sum(~(np.isnan(x) | np.isnan(agree))))
        alone = int(np.sum(single & ~np.isnan(agree)))
        note = (f"{drawn} genes; {alone} rest on a single guide, so nothing "
                f"corroborates them and they are indistinguishable from "
                f"agreement on a volcano.")
        if beyond:
            plural = beyond != 1
            note += (f" {beyond} gene{'s' if plural else ''} with far more "
                     f"guides than the rest of the library "
                     f"{'are' if plural else 'is'} drawn beyond the opening "
                     f"view; Reset view reaches {'them' if plural else 'it'}.")
        if self._has_usable_keys():
            note += " Click a gene for its coefficient."
        self.set_status(note)
        if self._selected_key is not None:
            self.highlight_key(self._selected_key)
        return drawn

    def _detail(self, index: int) -> str:
        frame = self._support
        if frame is None or not len(frame):
            return ""
        frame = frame.reset_index() if frame.index.name else frame
        if not 0 <= int(index) < len(frame):
            return ""
        row = frame.iloc[int(index)]
        parts = []
        if "n_same_direction" in frame and "n_guides" in frame:
            parts.append(f"{int(row['n_same_direction'])} of "
                         f"{int(row['n_guides'])} guides agree")
        if "gene_p" in frame and np.isfinite(row["gene_p"]):
            parts.append(f"gene p = {row['gene_p']:.3g}")
        if "single_guide" in frame and bool(row["single_guide"]):
            parts.append("SINGLE GUIDE -- gene p IS that guide's p")
        return "   ".join(parts)


class ResultsTable(QWidget):
    """The coefficient table, sortable and searchable, wired to a plot.

    A volcano answers "which points are extreme"; it cannot answer "what is
    the q-value of TGGT1_233460_4". Reading numbers off a scatter is the wrong
    tool, and until now the only way to see them was to open the CSV.

    :ivar row_selected: emitted with the frame row index of the selected row.
    :ivar key_selected: emitted with the selected row's identifier. This is
        the one to connect a plot to: an index only means anything to the
        frame it came from, and the table's frame and the plot's frame are
        not required to be the same one.
    """

    row_selected = Signal(int)
    key_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        from PySide6.QtWidgets import (QAbstractItemView, QLineEdit,
                                       QTableWidget)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        top = QHBoxLayout()
        self._filter = QLineEdit()
        self._filter.setPlaceholderText(
            "Filter rows — type a gene, a guide, anything in the table")
        self._filter.textChanged.connect(self._on_filter_text)
        top.addWidget(self._filter, 1)
        self._only_hits = QCheckBox("significant only")
        self._only_hits.toggled.connect(self._apply_filter)
        top.addWidget(self._only_hits)
        self._copy = QPushButton("Copy")
        self._copy.setToolTip("Copy the visible rows as TSV.")
        self._copy.clicked.connect(self.copy_visible)
        top.addWidget(self._copy)
        layout.addLayout(top)

        self.table = QTableWidget(0, 0)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.itemSelectionChanged.connect(self._on_selection)
        layout.addWidget(self.table, 1)

        self._count = QLabel("")
        layout.addWidget(self._count)

        self._frame = None
        self._alpha = 0.05
        self._key_column: Optional[str] = None
        # EVERY PIECE OF STATE _apply_filter READS IS BORN HERE.
        #
        # `_significance` was created only in set_frame, and the filter
        # controls are connected in this constructor -- so any path that
        # touched a control before the first frame arrived crashed the
        # application on startup with AttributeError. configure() is one such
        # path: it can uncheck "significant only", which emits toggled.
        #
        # A widget must be fully usable the moment it exists. Half-built state
        # that only becomes valid after some other method has been called is
        # how a constructor turns into a trap.
        self._significance: Optional[str] = None
        #: Identifiers the table has been narrowed to from a plot, or None.
        self._key_restriction: Optional[set] = None

    def set_frame(self, frame, *, alpha: float = 0.05,
                  significance_column: Optional[str] = None,
                  key_column: Optional[str] = None) -> int:
        """Fill the table. Returns the row count."""
        from PySide6.QtWidgets import QTableWidgetItem

        self._frame = frame
        self._alpha = alpha
        # A new table is a new experiment: a set of keys chosen off the last
        # one names nothing here, and leaving it on would hide every row.
        self._key_restriction = None
        self._key_column = key_column or (
            "feature" if frame is not None and "feature" in frame.columns
            else None)
        self._significance = significance_column or self._guess_significance(frame)
        if frame is None or not len(frame):
            self.table.setRowCount(0)
            self._count.setText("Nothing to show.")
            return 0

        columns = list(frame.columns)
        # Sorting must be off while filling: with it on, Qt re-sorts after
        # every insert and the rows end up interleaved.
        self.table.setSortingEnabled(False)
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.setRowCount(len(frame))
        for row in range(len(frame)):
            for column, name in enumerate(columns):
                value = frame.iloc[row][name]
                item = _NumericItem(value)
                # The frame row, so a click still maps home after sorting.
                item.setData(Qt.UserRole, row)
                self.table.setItem(row, column, item)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()
        self._apply_filter()
        return len(frame)

    @staticmethod
    def _guess_significance(frame) -> Optional[str]:
        """Prefer a corrected column: filtering on raw p would mislead."""
        if frame is None:
            return None
        for name in ("q_value", "adjusted_p_value", "p_value"):
            if name in frame.columns:
                return name
        return None

    def show_keys(self, keys) -> int:
        """Narrow the table to a set of identifiers. ``None`` clears it.

        The other end of :attr:`FastPlot.keys_selected`: a histogram bar is a
        hundred coefficients and cannot select one of them, but "show me the
        hundred" is a question the table can answer exactly. Returns how many
        rows are visible afterwards.
        """
        self._key_restriction = None if keys is None else {
            str(key) for key in keys}
        self._apply_filter()
        return sum(not self.table.isRowHidden(row)
                   for row in range(self.table.rowCount()))

    def _on_filter_text(self) -> None:
        """Typing is a new intent, so it drops a set chosen on a plot.

        Otherwise the two filters AND together and the user types a gene they
        can see in the plot, gets nothing, and has no way to find out why.
        """
        self._key_restriction = None
        self._apply_filter()

    def _apply_filter(self) -> None:
        text = self._filter.text().strip().lower()
        hits_only = self._only_hits.isChecked()
        # The significance cut needs the frame to find its column in. Without
        # one there is nothing to cut on, and asking for the column would be
        # the same crash one line further down.
        significance = self._significance if self._frame is not None else None
        shown = 0
        for row in range(self.table.rowCount()):
            visible = True
            if text:
                visible = any(
                    text in (self.table.item(row, c).text() or "").lower()
                    for c in range(self.table.columnCount())
                    if self.table.item(row, c) is not None)
            if visible and hits_only and significance:
                if significance not in self._frame.columns:
                    continue
                column = list(self._frame.columns).index(significance)
                item = self.table.item(row, column)
                try:
                    visible = float(item.text()) <= self._alpha
                except (TypeError, ValueError):
                    visible = False
            if visible and self._key_restriction is not None:
                item = self.table.item(row, 0)
                index = None if item is None else item.data(Qt.UserRole)
                key = None if index is None else self.key_for_row(int(index))
                visible = key is not None and key in self._key_restriction
            self.table.setRowHidden(row, not visible)
            shown += int(visible)
        total = self.table.rowCount()
        note = f"{shown} of {total} rows"
        if hits_only and significance:
            note += f" ({significance} <= {self._alpha:g})"
        if self._key_restriction is not None:
            # Said out loud, because a table that has silently narrowed itself
            # is indistinguishable from a table that has lost its rows.
            note += (f" — narrowed to {len(self._key_restriction)} chosen on "
                     f"a plot; type here to clear")
        self._count.setText(note)

    def _on_selection(self) -> None:
        items = self.table.selectedItems()
        if not items:
            return
        index = items[0].data(Qt.UserRole)
        if index is None:
            return
        index = int(index)
        self.row_selected.emit(index)
        key = self.key_for_row(index)
        if key is not None:
            self.key_selected.emit(key)

    def configure(self, *, placeholder: Optional[str] = None,
                  significance_filter: Optional[bool] = None) -> None:
        """Adapt the controls to what the table actually holds.

        This widget is reused for tables that are not coefficient tables --
        the sweep's runs, for one -- and a filter offering "significant only"
        over a list of trials is a control that cannot do anything, sitting
        next to a placeholder telling the user to type a gene into it.
        """
        if placeholder is not None:
            self._filter.setPlaceholderText(placeholder)
        if significance_filter is not None:
            self._only_hits.setVisible(bool(significance_filter))
            if not significance_filter:
                self._only_hits.setChecked(False)

    def key_for_row(self, index: int) -> Optional[str]:
        """The identifier at frame position ``index``, or ``None``."""
        if self._frame is None or not self._key_column:
            return None
        if self._key_column not in self._frame.columns:
            return None
        if not 0 <= int(index) < len(self._frame):
            return None
        return str(self._frame[self._key_column].iloc[int(index)])

    def select_frame_row(self, index: int) -> bool:
        """Scroll to and select the row for frame position ``index``.

        This is the other half of clicking a point on the volcano: the dot and
        the numbers behind it should be two views of one thing.
        """
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is not None and item.data(Qt.UserRole) == index:
                self.table.selectRow(row)
                self.table.scrollToItem(item)
                return True
        return False

    def select_key(self, key) -> bool:
        """Select the row whose identifier is ``key``. The safe direction.

        A plot has no business knowing where a row sits in this table -- the
        user sorts it, filters it, and after instruction 122 it may not even
        be drawn from the same frame. It knows the key, and the key is enough.

        A hidden row is unhidden to select it: silently doing nothing because
        the filter box excludes the point the user just clicked reads as a
        broken click.
        """
        if self._frame is None or not self._key_column:
            return False
        if self._key_column not in self._frame.columns:
            return False
        wanted = str(key)
        column = list(self._frame.columns).index(self._key_column)
        for row in range(self.table.rowCount()):
            item = self.table.item(row, column)
            if item is not None and item.text() == wanted:
                self.table.setRowHidden(row, False)
                self.table.selectRow(row)
                self.table.scrollToItem(item)
                return True
        return False

    def copy_visible(self) -> str:
        """Put the visible rows on the clipboard as TSV, and return them."""
        from PySide6.QtWidgets import QApplication

        lines = ["\t".join(
            self.table.horizontalHeaderItem(c).text()
            for c in range(self.table.columnCount()))]
        for row in range(self.table.rowCount()):
            if self.table.isRowHidden(row):
                continue
            lines.append("\t".join(
                (self.table.item(row, c).text() if self.table.item(row, c)
                 else "")
                for c in range(self.table.columnCount())))
        text = "\n".join(lines)
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
        return text


try:  # pragma: no cover - trivial subclass
    from PySide6.QtWidgets import QTableWidgetItem

    def _is_missing(value) -> bool:
        """True for None, NaN, pandas' NA and NaT.

        `pandas.isna` rather than `value != value`, which was the first
        attempt and got pd.NA exactly wrong: `pd.NA != pd.NA` evaluates to
        pd.NA, not to True, and `bool(pd.NA)` RAISES -- so the except branch
        swallowed it and reported the one sentinel this was written for as
        present.

        Importing pandas here costs nothing: every value this sees came out
        of a DataFrame handed to `set_frame`, so pandas is already imported
        by definition of having anything to render.
        """
        if value is None:
            return True
        try:
            import pandas as _pd

            return bool(_pd.isna(value))
        except (ImportError, TypeError, ValueError):
            # isna on an array-like returns an array; a cell holding one is
            # not missing.
            return False

    class _NumericItem(QTableWidgetItem):
        """Sorts numerically when it holds a number, textually otherwise.

        A plain QTableWidgetItem sorts "10" before "9", which on a q-value
        column puts the answer in the wrong place.
        """

        def __init__(self, value):
            # EVERY FLAVOUR OF MISSING SHOWS AS EMPTY. `str(pd.NA)` is the
            # literal "<NA>" and `str(float('nan'))` is "nan", so a column
            # that is absent for some rows -- a session run has no trial_id,
            # a running one has no result count -- printed those words into
            # the table as though they were data. An empty cell is what a
            # missing value looks like everywhere else in this application.
            super().__init__("" if _is_missing(value) else str(value))
            try:
                self._number = float(value)
            except (TypeError, ValueError):
                self._number = None

        def __lt__(self, other):
            mine = getattr(self, "_number", None)
            theirs = getattr(other, "_number", None)
            if mine is not None and theirs is not None:
                return mine < theirs
            return self.text() < other.text()
except Exception:  # pragma: no cover
    _NumericItem = None
