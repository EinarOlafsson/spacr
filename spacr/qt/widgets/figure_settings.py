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
from json import JSONDecodeError
from typing import Callable, Optional

LOG = logging.getLogger(__name__)

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import (
    QCheckBox,
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

from .colour_picker import pick_colour

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
        # Qt's own dialog, never the platform one -- see
        # :mod:`spacr.qt.widgets.colour_picker`.
        colour = pick_colour(button, state["colour"])
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
        # The per-figure text size is an ATTRIBUTE, and `reject` restores the
        # figure by copying axes out of the snapshot rather than by swapping
        # the object -- so the attribute would survive a Cancel that undid
        # everything it applies to. Kept here and put back explicitly.
        try:
            from .figure_queue import figure_text_size_override
            self._text_size_at_open = figure_text_size_override(figure)
        except Exception:  # pragma: no cover - figure_queue unavailable
            self._text_size_at_open = 0

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
        # STATISTICS, only for a figure that actually compares groups. A tab
        # offering a t-test on a Q-Q plot would be an invitation to report a
        # number that means nothing -- see `_statistics_tab`.
        if getattr(figure, "_spacr_groups", None):
            self.tabs.addTab(self._scroll(self._statistics_tab()),
                             "Statistics")
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
        try:
            from .figure_queue import set_figure_text_size_override
            set_figure_text_size_override(
                self._figure, getattr(self, "_text_size_at_open", 0))
        except Exception:  # pragma: no cover - figure_queue unavailable
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

    def _statistics_tab(self) -> QWidget:
        """Which test, at what level, corrected how -- and what it chose.

        Asked for on 2026-08-16: "provide a statistics tab in the settings
        for the graph upon right clicking".

        ONLY OFFERED WHERE THERE IS SOMETHING TO COMPARE. The figure carries
        its groups or it does not; a tab offering a t-test on a Q-Q plot
        would invite a number that means nothing.

        The default is AUTO, and the panel shows what auto chose and why.
        Forcing a test is offered because a reader sometimes has a reason the
        data cannot express -- a paired design, a pre-registered analysis --
        but it is a deliberate override rather than the starting point, and
        the automatic choice is the one that reads the assumption checks.
        """
        from ...figures import stats as stats_module

        page = QWidget()
        form = QFormLayout(page)
        groups = dict(getattr(self._figure, "_spacr_groups", {}) or {})
        self._stats_state = {"test": None, "alpha": 0.05,
                             "correction": "fdr_bh", "unit": "coefficient"}

        form.addRow(QLabel(", ".join(
            f"{label} (n={len(values)})" for label, values in groups.items())))

        test = QComboBox()
        test.addItem("automatic — chosen from the data", None)
        for name in ("Student's t", "Welch's t", "Mann-Whitney U",
                     "one-way ANOVA", "Welch's ANOVA", "Kruskal-Wallis",
                     "paired t", "Wilcoxon signed-rank"):
            test.addItem(name, name)
        test.setToolTip(
            "Automatic reads the group count, a Levene test for equal "
            "variance and a Shapiro-Wilk for normality — and treats a check "
            "it had too few points to run as FAILED, because 'did not "
            "reject' on n=3 is not 'the assumption holds'.")
        form.addRow("Test", test)

        alpha = QDoubleSpinBox()
        alpha.setDecimals(3)
        alpha.setRange(0.001, 0.5)
        alpha.setSingleStep(0.005)
        alpha.setValue(0.05)
        form.addRow("Alpha", alpha)

        correction = QComboBox()
        try:
            from ...multiple_testing import METHODS

            for key in METHODS:
                correction.addItem(key, key)
            correction.setCurrentText("fdr_bh")
        except Exception:              # pragma: no cover - module absent
            correction.addItem("fdr_bh", "fdr_bh")
        correction.setToolTip(
            "Applied ACROSS the comparisons on this panel. Six pairwise "
            "tests at 0.05 is a 26% chance of one false positive, and the "
            "individual p-values give no hint of it.")
        form.addRow("Correct across pairs", correction)

        unit = QLineEdit("coefficient")
        unit.setToolTip(
            "What ONE observation is. spaCR measures thousands of cells "
            "across a handful of wells: a test across cells when the "
            "replicate is the well returns p < 1e-10 on pure noise, and "
            "nothing in the number itself says so.")
        form.addRow("Unit of replication", unit)

        verdict = QLabel("")
        verdict.setWordWrap(True)
        verdict.setStyleSheet("color: palette(mid); font-size: 11px;")
        form.addRow(verdict)

        def _recompute():
            self._stats_state.update(
                test=test.currentData(), alpha=float(alpha.value()),
                correction=correction.currentData() or "fdr_bh",
                unit=unit.text().strip() or "observation")
            lines = []
            labels = list(groups)
            for index, left in enumerate(labels):
                for right in labels[index + 1:]:
                    try:
                        result = stats_module.compare(
                            {left: groups[left], right: groups[right]},
                            unit=self._stats_state["unit"],
                            force=self._stats_state["test"])
                    except ValueError as refusal:
                        lines.append(f"{left} vs {right}: {refusal}")
                        continue
                    lines.append(f"{left} vs {right} — {result.sentence()}")
            verdict.setText("\n".join(lines) or "nothing to compare")

        for control in (test, correction):
            control.currentIndexChanged.connect(lambda *_: _recompute())
        alpha.valueChanged.connect(lambda *_: _recompute())
        unit.editingFinished.connect(_recompute)
        _recompute()

        self._stats_verdict = verdict
        return page

    def _figure_tab(self) -> QWidget:
        from .figure_queue import set_figure_text_size_override

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

        # One control that reaches EVERY text object at once, because "make
        # the fonts bigger" is a single intention.
        #
        # GitHub issue #108 (2026-08-17): "Font size is by default to large to
        # be visible. Adjusting font size ... from 10 to 2, does not reduce
        # the font size, in fact increases it, and when returning ... the font
        # size has been returned to 10."
        #
        # All three symptoms were this control, and the cause is what it did
        # NOT reach. Measured on a volcano-shaped figure: 23 text objects, 20
        # reached, and the three it missed were
        #
        #     ('EAF1', 22.0)      an ax.texts annotation -- a GENE LABEL, and
        #                         the LARGEST text on the figure
        #     ('a run', 12.0)     the figure suptitle
        #     ('condition', 10.0) the legend's title
        #
        # So shrinking "all text" shrank everything EXCEPT the biggest thing
        # on the plot, which then dominated it -- and reads exactly as "the
        # font got bigger". The volcano annotates its hits by name, so this
        # is the common case, not a corner one.
        all_text = QSpinBox()
        # Down to 2, because 2 is what the reporter typed. A 2pt font is
        # unreadable and that is their business; a control that silently
        # clamps is one that lies about what it did.
        all_text.setRange(2, 96)
        all_text.setValue(_current_text_size(figure))

        def set_all_text(size):
            for item in _every_text(figure):
                item.set_fontsize(size)
            # AND REMEMBER IT ON THE FIGURE. Setting the sizes alone was not
            # enough and that is issue #108's third symptom: the next full
            # render calls `render_figure_to_png`, which re-applies the
            # GLOBAL text-size preference to every text object, so the user's
            # choice survived only until the dialog closed -- and reopening
            # the dialog, which reads the size off the figure, showed the
            # preference again. The override is per FIGURE and is not written
            # to the preference, for the same reason the colour buttons on
            # this tab are not: this dialog restyles the figure in front of
            # the user, and the setting for every figure is Preferences.
            #
            # Connected AFTER `setValue` above, so this runs only when a user
            # moves the control -- seeding never writes back. That is the
            # rule at the head of the figure colour section in
            # `spacr/qt/preferences.py`: NEVER PERSIST A RESOLVED DEFAULT.
            set_figure_text_size_override(figure, size)
            self._changed()
        all_text.valueChanged.connect(set_all_text)
        all_text.setToolTip(
            "The size of every piece of text in this figure: the title, the "
            "axis labels, the tick labels, the legend and any annotation. It "
            "is remembered for this figure, so a redraw keeps it. The size "
            "every figure starts at is in Preferences → Figures.")
        form.addRow("All text size", all_text)

        # AND THE COLOUR OF ALL OF IT -- IN TWO CONTROLS, NOT ONE.
        #
        # There was a size control here and no colour control at all, so the
        # background could be changed and the writing on top of it could not,
        # which on a dark background is a figure with invisible axes and no
        # way to fix it. The first version of the fix was ONE "All text
        # colour" that also drove the spines and the tick marks.
        #
        # The maintainer's decision (instruction 152 B) splits it by what a
        # mark IS rather than by which code draws it: "line color which
        # should change the color of all lines including axis lines and
        # ticks, and then a font color that controls the color of all font in
        # the graph". So a user can now say "dark axes, coloured labels" or
        # the other way round, and the first report -- "doesnt look like
        # there is an option to change the axis color" -- has an answer that
        # is not "change your text as well".
        def set_line_ink(colour):
            apply_line_colour(figure, colour)
            self._changed()

        def set_font_ink(colour):
            apply_font_colour(figure, colour)
            self._changed()

        current_font_colour = "#000000"
        current_line_colour = "#000000"
        if figure.axes:
            try:
                current_font_colour = _as_hex(
                    figure.axes[0].xaxis.label.get_color())
            except Exception:      # pragma: no cover - odd colour spec
                pass
            try:
                spines = list(figure.axes[0].spines.values())
                if spines:
                    current_line_colour = _as_hex(spines[0].get_edgecolor())
            except Exception:      # pragma: no cover - odd colour spec
                current_line_colour = current_font_colour
        form.addRow("Line colour",
                    _colour_button(current_line_colour, set_line_ink))
        form.addRow("Font colour",
                    _colour_button(current_font_colour, set_font_ink))

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


def figure_line_artists(figure) -> list:
    """Every LINE on ``figure`` that the line control reaches.

    The data's own lines, the reference and threshold lines, the trends and
    the Q-Q diagonal -- all of them ``Line2D`` on an axes -- plus the axis
    SPINES, plus a legend's sample lines so the key does not go on describing
    the colour the figure no longer uses.

    THE TICK MARKS ARE NOT HERE, and not because they are exempt: they have
    no artist to hand back. matplotlib draws them from the tick's own pen, so
    :func:`apply_line_colour` reaches them through ``tick_params``. This
    function exists to be counted and asserted against, and a count that
    silently omitted the ticks would make that assertion a lie.

    GRIDLINES ARE EXCLUDED. A grid repainted in the ink is a cage over the
    data; :data:`spacr.figure_style.PRINT_GRID` says the same thing about the
    save path. They are not in ``axes.lines`` either, so this is a statement
    of intent rather than a filter.
    """
    found = []
    for axis in getattr(figure, "axes", ()):
        found += list(getattr(axis, "lines", ()))
        found += list(axis.spines.values())
        legend = axis.get_legend()
        if legend is not None:
            found += list(legend.get_lines())
    return found


def apply_line_colour(figure, colour) -> int:
    """Draw every line on ``figure`` in ``colour``. Returns how many.

    EVERY LINE MEANS THE AXES TOO -- the spines and the
    tick marks, which is the half that had no control at all and which the
    first report named. The same division as
    :meth:`spacr.qt.widgets.fast_plots.FastPlot.set_line_style` makes on the
    pyqtgraph side, so a figure looks the same whichever engine drew it.

    THE DASH PATTERNS SURVIVE, because only ``set_color`` is called: a
    threshold line stays dashed and the reference line stays solid, which is
    what tells a reader which is which. Rebuilding the line would flatten
    that on every restyle -- the pyqtgraph half copies its pen for exactly
    this reason.
    """
    touched = 0
    for artist in figure_line_artists(figure):
        try:
            if hasattr(artist, "set_edgecolor"):
                artist.set_edgecolor(colour)     # a spine
            else:
                artist.set_color(colour)
            touched += 1
        except Exception:                        # pragma: no cover - odd spec
            continue
    for axis in getattr(figure, "axes", ()):
        # THE TICK MARKS, SEPARATELY, and `color=` only. `colors=` would set
        # the LABEL as well, which is the conflation the two controls exist
        # to undo -- and it is done through `tick_params` rather than over
        # the current ticks because matplotlib rebuilds them on every draw,
        # so a colour set on the objects is lost at the next autoscale.
        try:
            axis.tick_params(color=colour, which="both")
        except Exception:                        # pragma: no cover
            continue
    return touched


def apply_font_colour(figure, colour) -> int:
    """Draw every piece of text on ``figure`` in ``colour``. Returns how many.

    EVERY PIECE MEANS THE TICK LABELS, the gene
    annotations, the suptitle and the legend's title -- the three that
    :func:`_every_text` exists to not miss, because a control called "all
    text" that reaches twenty of twenty-three objects reads as broken rather
    than as incomplete (issue #108).
    """
    touched = 0
    for item in _every_text(figure):
        try:
            item.set_color(colour)
            touched += 1
        except Exception:                        # pragma: no cover
            continue
    for axis in getattr(figure, "axes", ()):
        # The labels are regenerated on every draw, so the colour has to be
        # set on the TICK rather than only on today's label objects.
        try:
            axis.tick_params(labelcolor=colour, which="both")
        except Exception:                        # pragma: no cover
            continue
    return touched


def figure_follows_the_theme(figure) -> None:
    """Put both colours back to what the app theme and the preferences say.

    The way out of a colour, and it has to exist: the design is a
    preference that froze because a resolved default was written back over
    the word "auto", and a per-figure control a user can only ever SET is
    that same freeze performed by hand.

    Reads the preference rather than remembering what the figure was drawn
    with, and that is the honest direction here: a matplotlib figure arrives
    already themed by
    :func:`spacr.qt.widgets.figure_queue._style_figure_colors`, so "the
    colour it was drawn with" IS the preference's answer -- unlike the
    pyqtgraph side, where each line carries a palette colour of its own worth
    restoring (``_spacr_base_colour``).
    """
    try:
        from ..preferences import get_figure_colors, get_figure_line_colour

        _bg, font = get_figure_colors()
        line = get_figure_line_colour()
    except Exception:                            # pragma: no cover - no store
        font = line = "#000000"
    apply_line_colour(figure, line)
    apply_font_colour(figure, font)


#: What a saved graph-style file says it is. A file that does not say this is
#: refused rather than partially applied, for the reason
#: `fast_plots.load_style` gives about a style of the WRONG KIND: settings
#: whose names happen to match would be taken and the rest left, which looks
#: like a corrupted house style rather than like a mistake.
GRAPH_STYLE_FILE_KIND = "spacr_graph_style"


def graph_style_as_dict(general=None, per_graph=None) -> dict:
    """The user's graph style as the thing that gets written to a file.

    This supplies save, load, and per-project defaults for figures that have no
    style
    DATACLASS, which is nearly all of them.

    It uses the existing style vocabulary. Capturing a second set of
    appearance keys from the artists of one drawn figure would create a third
    style system. The deltas here are what
    :func:`spacr.figures.style.user_overrides` and
    :func:`spacr.figure_style.resolve` already read,
    so a loaded house style reaches every figure spaCR draws without anything
    else being wired.

    It stores no additional colours. The tempting alternative -- "capture what this figure
    looks like right now" -- would sample the ink the THEME resolved. Saving
    that and applying it later would write back a resolved default that stays
    invisible until the first time the user changes
    theme. The live figure's ink stays a token in `prefs/figure_*`.
    """
    if general is None or per_graph is None:
        try:
            from ..preferences import (get_figure_style,
                                       get_figure_style_per_graph)
            if general is None:
                general = get_figure_style()
            if per_graph is None:
                per_graph = get_figure_style_per_graph()
        except Exception:                    # pragma: no cover - no store
            general, per_graph = general or {}, per_graph or {}
    return {
        "spacr_style_kind": GRAPH_STYLE_FILE_KIND,
        "general": dict(general or {}),
        "per_graph": {str(kind): dict(values)
                      for kind, values in (per_graph or {}).items()
                      if isinstance(values, dict)},
    }


def save_graph_style(path: str, general=None, per_graph=None) -> str:
    """Write the graph style to ``path`` as JSON. Returns the path, or "".

    The DELTAS, exactly as the store holds them, so a house style saved on a
    machine whose package defaults have since improved still means "these
    four things differ" rather than freezing the defaults of the day it was
    written. Same reason `FigureStylePreferences.values` returns deltas.
    """
    import json

    if not path:
        return ""
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(graph_style_as_dict(general, per_graph), handle,
                      indent=2, sort_keys=True)
    except OSError as error:
        LOG.warning("could not save the graph style to %s: %s", path, error)
        return ""
    return path


def load_graph_style(path: str) -> tuple:
    """``(general, per_graph)`` from a saved graph style.

    :raises ValueError: if the file is not a spaCR graph style. Refused
        rather than partially applied -- see :data:`GRAPH_STYLE_FILE_KIND`.

    FORWARDS-COMPATIBLE IN BOTH DIRECTIONS. A key the package no longer has
    is KEPT, not dropped: `FigureStylePreferences` already shows such a value
    as "<value> (not offered)" rather than snapping it to something else, and
    a loader that discarded it would make opening and re-saving a colleague's
    house style silently lose the parts this build does not know about.
    """
    import json

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or \
            data.get("spacr_style_kind") != GRAPH_STYLE_FILE_KIND:
        raise ValueError(f"{path} is not a spaCR graph style")
    general = data.get("general")
    per_graph = data.get("per_graph")
    return (dict(general) if isinstance(general, dict) else {},
            {str(kind): dict(values)
             for kind, values in (per_graph or {}).items()
             if isinstance(values, dict)})


def apply_graph_style(general, per_graph) -> None:
    """Make a loaded graph style THIS PROJECT'S DEFAULT.

    Written into the same preference the Figures tab writes, which is what
    makes "applied to every figure of that type without re-setting it each
    time" true: `figure_style.resolve` and `figures.style.user_overrides`
    both read it already. Loading a style is therefore indistinguishable
    afterwards from having set every one of its controls by hand, which is
    the property that stops this being a fourth place a setting can live.
    """
    from ..preferences import set_figure_style, set_figure_style_per_graph

    set_figure_style(dict(general or {}))
    set_figure_style_per_graph({str(kind): dict(values)
                                for kind, values in (per_graph or {}).items()
                                if isinstance(values, dict)})


def add_graph_style_file_entries(menu, parent=None, *, on_change=None) -> None:
    """"Save graph style…" / "Load graph style…" on ``menu``.

    This makes matplotlib figures editable and savable. `fast_plots.save_style` and
    `load_style` already do this for a style DATACLASS, and the only
    dataclass in the package is `VolcanoStyle`, so on 2026-08-18 the savable
    half existed and nothing a user could click reached it.
    """
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    owner = parent if parent is not None else menu

    def _save():
        path, _filter = QFileDialog.getSaveFileName(
            owner, "Save graph style", "graph_style.json",
            "spaCR graph style (*.json);;All files (*)")
        if path:
            save_graph_style(path)

    def _load():
        path, _filter = QFileDialog.getOpenFileName(
            owner, "Load graph style", "",
            "spaCR graph style (*.json);;All files (*)")
        if not path:
            return
        try:
            general, per_graph = load_graph_style(path)
        except (OSError, ValueError, JSONDecodeError) as error:
            QMessageBox.warning(owner, "Load graph style", str(error))
            return
        apply_graph_style(general, per_graph)
        if callable(on_change):
            try:
                on_change(preview=True)
            except TypeError:
                on_change()

    save = QAction("Save graph style…", owner)
    save.setToolTip(
        "Write the general and per-graph settings from Preferences to a "
        "file, so a lab's house style can be shared and re-applied.")
    save.triggered.connect(_save)
    menu.addAction(save)

    load = QAction("Load graph style…", owner)
    load.setToolTip(
        "Read a saved house style and make it this project's default, so "
        "every figure drawn from now on uses it.")
    load.triggered.connect(_load)
    menu.addAction(load)


#: The plot types `create_grouped_plot` can draw the same data as, in the
#: order the menu offers them. Asked for by name 2026-08-19: "line, bar,
#: jitter-bar, jitter-box, jitter, box, violin".
GROUPED_PLOT_TYPES = (
    ("line", "Line"),
    ("bar", "Bar"),
    ("jitter_bar", "Jitter over bar"),
    ("jitter_box", "Jitter over box"),
    ("jitter", "Jitter"),
    ("box", "Box"),
    ("violin", "Violin"),
)


def _replot(figure, kind: str, on_change=None):
    """Redraw ``figure`` in place as ``kind``. Returns whether it changed.

    A NEW Figure, and that is not a choice: `create_grouped_plot` builds its
    own -- spacrGraph makes one and draws into it -- so there is nothing to
    draw "in place" onto. The caller is handed the new one through
    ``on_change`` and is responsible for putting it where the old one was;
    `FigureQueue.replace_figure` is what does that.

    Never raises: a plot type that cannot show this data is a menu entry that
    does nothing visible, not a crash in a right-click.

    :param on_change: called with the NEW figure when it is drawn. A callable
        taking no arguments is still accepted, for the toggles that only need
        telling that something moved.
    :returns: the new Figure, or ``None``.
    """
    recipe = dict(getattr(figure, "_spacr_replot", None) or {})
    if recipe.get("df") is None:
        return None
    try:
        from ...plot import create_grouped_plot

        recipe["graph_type"] = str(kind)
        drawn, _results = create_grouped_plot(save=False, **recipe)
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not redraw the figure as %r", kind, exc_info=True)
        return None
    if drawn is None:
        return None
    drawn._spacr_replot = recipe
    if callable(on_change):
        try:
            on_change(drawn)
        except TypeError:
            try:
                on_change()
            except Exception:                                # noqa: BLE001
                LOG.debug("redraw notification failed", exc_info=True)
        except Exception:                                    # noqa: BLE001
            LOG.debug("redraw notification failed", exc_info=True)
    return drawn


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

    # SHOW THE SAME DATA ANOTHER WAY (178 A). Offered only where the figure
    # carries its own recipe -- `create_grouped_plot` attaches one -- because
    # a menu entry that cannot redraw the figure it is on is worse than an
    # absent one. Every other figure in spaCR simply does not get the group.
    recipe = getattr(figure, "_spacr_replot", None)
    if isinstance(recipe, dict) and recipe.get("df") is not None:
        show_as = menu.addMenu("Show as")
        current = str(recipe.get("graph_type") or "")
        for kind, label in GROUPED_PLOT_TYPES:
            action = show_as.addAction(label)
            action.setCheckable(True)
            action.setChecked(kind == current)
            action.triggered.connect(
                lambda _checked=False, k=kind: _replot(figure, k, on_change))

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

    scales = QMenu("Axis scale", menu)      # see "Appearance" below for why
    menu.addMenu(scales)
    for name, setter in (("X", "set_xscale"), ("Y", "set_yscale")):
        submenu = QMenu(name, scales)
        scales.addMenu(submenu)
        for scale in AXIS_SCALES:
            action = QAction(scale, parent)
            action.triggered.connect(
                lambda _checked=False, s=scale, m=setter:
                _apply(lambda a: getattr(a, m)(s)))
            submenu.addAction(action)

    menu.addSeparator()
    # THE TWO COLOUR CONTROLS, ON THE RIGHT-CLICK ITSELF (instruction 152 B).
    # They are two clicks away behind "Figure settings…", and the report that
    # opened 152 was a user who could not find an axis colour at all -- a
    # control nobody can find is a control that does not exist.
    # BUILT WITH AN EXPLICIT PARENT, not `menu.addMenu("Appearance")`.
    # `addMenu(str)` hands back a QMenu that PySide does not keep alive: the
    # Python wrapper is the only owner, and the moment it goes out of scope
    # the C++ object is deleted under the still-visible parent action. Driving
    # the entry then raises "Internal C++ object (QMenu) already deleted",
    # which is what a user would see as a submenu that opens empty.
    appearance = QMenu("Appearance", menu)
    menu.addMenu(appearance)

    def _pick_ink(title, apply_to):
        current = "#000000"
        try:
            if axes:
                current = _as_hex(axes[0].xaxis.label.get_color())
        except Exception:                        # pragma: no cover
            pass
        chosen = pick_colour(parent, current, title)
        if chosen.isValid():
            apply_to(figure, chosen.name())
            _notify()

    line_action = QAction("Line colour…", parent)
    line_action.setToolTip(
        "Every line in the figure, the axis spines and the tick marks "
        "included. The numbers beside the ticks are text and follow the "
        "font colour.")
    line_action.triggered.connect(
        lambda: _pick_ink("Line colour", apply_line_colour))
    appearance.addAction(line_action)

    font_action = QAction("Font colour…", parent)
    font_action.setToolTip(
        "Every piece of text in the figure: the title, the axis labels, the "
        "tick labels, the legend and any annotation.")
    font_action.triggered.connect(
        lambda: _pick_ink("Font colour", apply_font_colour))
    appearance.addAction(font_action)

    theme_action = QAction("Follow the theme (colours)", parent)
    theme_action.setToolTip(
        "Put both colours back to the app theme and the figure preferences.")
    theme_action.triggered.connect(
        lambda: (figure_follows_the_theme(figure), _notify()))
    appearance.addAction(theme_action)

    menu.addSeparator()
    save = QAction("Save figure as…", parent)
    save.setToolTip("Write this figure to a file, with the styling it has "
                    "on screen right now.")
    save.triggered.connect(lambda: save_figure_as(parent, figure))
    menu.addAction(save)

    # STYLE IT FOR THE FILE FIRST (178 C.2). "the user should be able to
    # change all of theis for the saved graph, get a preview then save."
    # Beside the direct save rather than replacing it: writing what is on
    # screen is one click and remains one click.
    styled = QAction("Save figure with a preview…", parent)
    styled.setToolTip(
        "Choose the ink, background, grid, size and resolution FOR THE FILE, "
        "see exactly what will be written, then write it. The figure on "
        "screen is not changed.")
    styled.triggered.connect(lambda: _open_styled_save(parent, figure))
    menu.addAction(styled)

    add_graph_style_file_entries(menu, parent, on_change=on_change)

    settings = QAction("Figure settings…", parent)
    if open_settings is not None:
        settings.triggered.connect(lambda: open_settings())
    menu.addAction(settings)
    return menu


def _every_text(figure):
    """Every text object on ``figure``, including the ones easily missed.

    ONE IMPLEMENTATION, in :func:`spacr.qt.widgets.figure_queue.figure_text_items`
    -- which carries the measurement and the reason. It lives there and not
    here because the RENDER pass needs it too and that pass runs on a worker
    thread with no Qt, and because the second half of issue #108 was these
    two reaching different sets of text: the dialog resized all twenty-three
    objects and the render put the global preference back over twenty of
    them. A copy here would drift again.
    """
    from .figure_queue import figure_text_items

    return figure_text_items(figure)


def _current_text_size(figure, default: int = 10) -> int:
    """The size the control should OPEN at: what the figure actually uses.

    It opened at a hardcoded 10 whatever the figure was set to, which is the
    third symptom of issue #108 -- "when returning to the Figure settings
    button menu the font size has been returned to 10". It had never left 10;
    it had never read the figure at all.

    The MOST COMMON size, not the largest or the mean: a figure has many tick
    labels at the body size and one or two headings above it, so the mode is
    what "the font size of this figure" means to a reader. Ties go to the
    smaller, so a figure with equal counts opens at its body size.
    """
    from collections import Counter

    sizes = []
    for item in _every_text(figure):
        try:
            if str(item.get_text()).strip():
                sizes.append(round(float(item.get_fontsize())))
        except Exception:                                        # noqa: BLE001
            continue
    if not sizes:
        return default
    counts = Counter(sizes)
    best = max(counts.values())
    return min(size for size, count in counts.items() if count == best)


def _open_styled_save(parent, figure):
    """Open the style-preview-save dialog. Returns it, or ``None``.

    Kept on the parent so Python does not collect it the moment this
    returns -- the same reason every other window this application opens is
    held somewhere.
    """
    if figure is None:
        return None
    try:
        from .save_figure_dialog import SaveFigureDialog

        dialog = SaveFigureDialog(figure, parent=parent)
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not open the styled save", exc_info=True)
        return None
    dialog.show()
    if parent is not None:
        kept = getattr(parent, "_spacr_save_dialogs", None)
        if kept is None:
            kept = []
            try:
                parent._spacr_save_dialogs = kept
            except Exception:                                # noqa: BLE001
                return dialog
        kept.append(dialog)
    return dialog


def save_figure_as(parent, figure, path: str = "") -> str:
    """Write ``figure`` to a file the user picks. Returns the path, or "".

    The design asks for a figure to be savable from the same right-click
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

    # THROUGH `spacr.plot.save_figure`, WHICH IS THE POINT OF INSTRUCTION 108
    # POINT 6. This function used to write the file itself, with the SCREEN's
    # background and no print rule, which made it one of the twenty-three
    # `savefig` calls that bypass the one place a figure the user keeps gets
    # written -- and on a dark theme it produced exactly instruction 150's
    # report: white text on a transparent ground, invisible the moment it is
    # pasted into a manuscript. `save_figure` applies the DPI preference and
    # `print_ready`'s ink rule, and a light-mode save is unchanged by it.
    #
    # THE USER'S OWN EXTENSION WINS over the format preference, and that is
    # the one thing this cannot delegate: `save_figure` corrects the extension
    # to the chosen FORMAT, so a user who typed `figure.pdf` while the
    # preference says PNG would get a PNG. The extension is passed as `fmt`
    # when it is one `save_figure` knows.
    extension = os.path.splitext(path)[1].lower().lstrip(".")
    try:
        from ...plot import FIGURE_FORMATS, print_ready, save_figure
    except Exception:                    # pragma: no cover - Qt-only build
        FIGURE_FORMATS, print_ready, save_figure = (), None, None

    if save_figure is not None and extension in FIGURE_FORMATS:
        try:
            return str(save_figure(figure, path, fmt=extension,
                                   bbox_inches="tight", close=False))
        except Exception as error:       # noqa: BLE001 - report, do not raise
            LOG.info("could not save figure to %s: %s", path, error)
            return ""
        finally:
            export_sidecars(figure, path)

    # SVG and EPS are NOT among `FIGURE_FORMATS`, so they cannot go through
    # `save_figure` without having their extension rewritten under them --
    # and they are offered in the dialog because they are what a journal asks
    # for. They still get the print rule, which is the half that matters.
    try:
        from ..preferences import (figure_bg_is_transparent, get_figure_colors,
                                   get_figure_png_dpi)
        background, _foreground = get_figure_colors()
        dpi = get_figure_png_dpi()
    except Exception:                    # pragma: no cover - no settings store
        background, dpi = "none", 200

        def figure_bg_is_transparent(value):
            return str(value).lower() in ("none", "transparent")

    from contextlib import nullcontext

    try:
        # Vector formats have no meaningful DPI, and passing one makes
        # matplotlib rasterise text in some backends.
        vector = extension in ("pdf", "svg", "eps")
        ink = print_ready(figure) if print_ready is not None else nullcontext()
        with ink:
            figure.savefig(
                path, bbox_inches="tight", facecolor=background,
                transparent=figure_bg_is_transparent(background),
                **({} if vector else {"dpi": dpi}))
    except Exception as error:           # noqa: BLE001 - report, do not raise
        LOG.info("could not save figure to %s: %s", path, error)
        return ""
    export_sidecars(figure, path)
    return path


def export_sidecars(figure, path) -> list:
    """Write the figure's DATA and its STATISTICS beside it.

    When a graph is exported, its data is also
    exported with the filename of the graph and a stats table is generated
    with the correct stats".

        volcano.pdf          the figure
        volcano.csv          the rows it actually drew
        volcano_stats.csv    the test, its assumptions, and its result

    One basename, so "where do these numbers come from" is answered by the
    folder rather than by the analyst's memory. That is the whole point: a
    figure that can go in a paper is one whose numbers can be checked.

    THE DATA IS WHAT WAS DRAWN, not what the panel was handed. A volcano is
    given 1,213 coefficients and draws 1,212 -- the nuisance terms are not
    hypotheses -- and a CSV whose row count disagrees with the n on the
    picture is worse than no CSV, because the CSV is what a reader believes.

    Never raises: an export that produced the figure has already done the
    useful part.

    :returns: the paths written.
    """
    written = []
    base = os.path.splitext(os.fspath(path))[0]

    frame = getattr(figure, "_spacr_data", None)
    if frame is not None:
        try:
            target = f"{base}.csv"
            frame.to_csv(target, index=False)
            written.append(target)
        except Exception as error:       # noqa: BLE001
            LOG.info("could not export the figure's data: %s", error)

    groups = getattr(figure, "_spacr_groups", None)
    if groups:
        try:
            from ...figures.stats import compare, table

            usable = {label: values for label, values in groups.items()
                      if values is not None and len(values) >= 2}
            if len(usable) >= 2:
                # EVERY PAIR, corrected across them. Six pairwise tests at
                # 0.05 is a 26% chance of one false positive and the
                # individual p-values give no hint of it.
                labels = list(usable)
                comparisons = []
                for index, left in enumerate(labels):
                    for right in labels[index + 1:]:
                        try:
                            comparisons.append(compare(
                                {left: usable[left], right: usable[right]},
                                unit="coefficient"))
                        except ValueError:
                            continue
                if comparisons:
                    target = f"{base}_stats.csv"
                    table(comparisons).to_csv(target, index=False)
                    written.append(target)
        except Exception as error:       # noqa: BLE001
            LOG.info("could not export the figure's statistics: %s", error)

    caption = getattr(figure, "_spacr_caption", "")
    if caption:
        try:
            target = f"{base}_legend.txt"
            with open(target, "w") as handle:
                handle.write(caption + "\n")
            written.append(target)
        except Exception as error:       # noqa: BLE001
            LOG.info("could not export the figure's legend: %s", error)
    return written


__all__ = ["FigureSettingsDialog", "build_figure_context_menu", "AXIS_SCALES",
           "GRAPH_STYLE_FILE_KIND", "graph_style_as_dict", "save_graph_style",
           "load_graph_style", "apply_graph_style",
           "add_graph_style_file_entries",
           "export_sidecars", "save_figure_as"]


# ---------------------------------------------------------------------------
# INSTRUCTION 118 -- FIGURE PREFERENCES: GENERAL, AND PER GRAPH TYPE
#
#   "in the general app preferences in the figure tab theere should be general
#    graph settings and specialized settings for al the possible different
#    sets of graphs"
#
# The MODEL for this already existed: `spacr.figure_style` holds
# GENERAL_DEFAULTS, GRAPH_DEFAULTS and `resolve`, and `spacr.figures.style`
# lays a user's deltas over the publication house style. What did not exist
# was any way to SET them -- the Figures tab held format, DPI, cache size and
# the dynamic switch, and nothing at all about how a plot looks.
#
# BUILT FROM `figure_style`'S OWN TABLES, not from a hand-written list. That
# is the same decision `add_style_entries` made for instruction 108 and for
# the same reason: a style gains a key, the panel gains a control, and the two
# cannot fall out of step. It also means this file never has to know what a
# volcano is.
#
# THE STORE HOLDS DELTAS, NOT THE RESOLVED STYLE, and that contract is older
# than this panel -- `get_figure_style` returns {} on a fresh install and
# `figures.style.user_overrides` returns only the keys the user MOVED, so a
# user who has never opened Preferences gets the published house style
# exactly. Writing the whole resolved style here would replace the house style
# for everybody, which is the same class of mistake as instruction 152 A's
# persisted resolution.
# ---------------------------------------------------------------------------

#: Closed sets for style keys whose values are a choice rather than a number.
#:
#: DECLARED HERE AND NOT IN `figure_style`, which is another territory this
#: session does not own. `spines` is derived from that module's own
#: SPINE_PRESETS rather than copied, so the one set that already exists as
#: data cannot drift; the rest are read off the comments beside their
#: defaults ("sem | sd | ci95 | none") and SHOULD move into the module as
#: metadata the next time it is opened. Noted rather than silently duplicated.
STYLE_CHOICES = {
    "palette": ("colorblind", "deep", "muted", "pastel", "bright", "dark"),
    "grid_style": tuple(style for style, _label in LINE_STYLES),
    "threshold_style": tuple(style for style, _label in LINE_STYLES),
    "reference_style": tuple(style for style, _label in LINE_STYLES),
    "format": ("pdf", "png", "svg"),
    "colormap": ("viridis", "plasma", "inferno", "magma", "cividis",
                 "coolwarm", "RdBu_r"),
    "bins": ("auto", "sturges", "fd", "scott", "sqrt"),
    "error_bars": ("sem", "sd", "ci95", "none"),
    "aspect": ("equal", "auto"),
}


def style_choices_for(name: str) -> tuple:
    """The closed set ``name`` may take, or ``()``."""
    if name == "spines":
        try:
            from ...figure_style import SPINE_PRESETS

            return tuple(SPINE_PRESETS)
        except Exception:               # pragma: no cover - import guard
            return ("all", "left_bottom", "none")
    return tuple(STYLE_CHOICES.get(name, ()))


#: The value a transparent ground is stored as. matplotlib's own spelling:
#: ``to_rgba("none")`` is ``(0, 0, 0, 0)``, and
#: :func:`spacr.figure_style.rc_params` already forwards ``background``
#: straight into ``figure.facecolor`` and ``axes.facecolor``, so this reaches
#: every figure drawn through the house style without
#: :mod:`spacr.figure_style` being touched -- it is another territory, and
#: the vocabulary it needed turned out to be one it already had.
TRANSPARENT_STYLE_GROUND = "none"

#: Style keys whose control offers "Transparent" beside the colour.
#:
#: The maintainer's restatement of instruction 118, 2026-08-16: "figures
#: should not have a background not black not white just transparent". A
#: plain colour button cannot say that -- every colour it can return is
#: opaque -- so the one key a transparent value MEANS anything for gets a
#: checkbox as well. Not `foreground`: invisible text is not a style.
TRANSPARENT_CAPABLE = ("background",)


def _looks_like_a_colour(value) -> bool:
    return isinstance(value, str) and value.startswith("#")


def _is_transparent_ground(value) -> bool:
    """Whether a stored style value means "no ground at all"."""
    return str(value).strip().lower() in ("none", "transparent", "")


def style_setting_label(name: str) -> str:
    """``grid_colour`` -> ``Grid colour``. British spelling is kept as the
    key spells it, because the key is what a user editing the INI sees."""
    return str(name).replace("_", " ").strip().capitalize()


class FigureStylePreferences(QWidget):
    """The Figures tab's GENERAL and PER-GRAPH style settings.

    Two levels, and the split is the whole design: a general
    style covers what every figure shares, and a per-graph style overrides it
    for one kind, because the settings that make a volcano readable are not
    the ones that make a plate heatmap readable. Changing the volcano's point
    size must not touch the heatmaps -- there is a test named for it.

    :param general: the user's stored general deltas.
    :param per_graph: the user's stored per-graph deltas, ``{kind: {...}}``.

    ONE KIND AT A TIME, chosen from a combo. Seven kinds times a dozen
    settings is eighty-odd rows, and a preferences tab that long is one
    nobody finds anything in -- which is the failure the report ("the graphs
    look pretty ugly") is downstream of, not a new one to introduce.
    """

    def __init__(self, general=None, per_graph=None, parent=None):
        super().__init__(parent)
        from ...figure_style import (GENERAL_DEFAULTS, GRAPH_DEFAULTS,
                                     GRAPH_KINDS)

        self._general_defaults = dict(GENERAL_DEFAULTS)
        self._graph_defaults = {kind: dict(values)
                                for kind, values in GRAPH_DEFAULTS.items()}
        self._kinds = tuple(GRAPH_KINDS)
        self._general = dict(general or {})
        self._per_graph = {str(kind): dict(values)
                           for kind, values in (per_graph or {}).items()
                           if isinstance(values, dict)}

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)

        heading = QLabel(
            "Applies to every figure. A graph type below can override any "
            "of it.")
        heading.setWordWrap(True)
        column.addWidget(heading)

        general_form = QFormLayout()
        general_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        column.addLayout(general_form)
        #: ``{name: (getter, default)}`` for the general layer.
        self._general_controls = {}
        for name, default in self._general_defaults.items():
            value = self._general.get(name, default)
            widget, getter, setter = self._control(name, value)
            self._general_controls[name] = (getter, setter, default)
            general_form.addRow(style_setting_label(name), widget)

        column.addSpacing(8)
        picker_row = QHBoxLayout()
        picker_row.addWidget(QLabel("Graph type"))
        self._kind_box = QComboBox()
        self._kind_box.setToolTip(
            "Settings for one kind of graph, laid over the general ones "
            "above. Only what you change here is stored, so a graph type you "
            "have not touched follows the general settings.")
        for kind in self._kinds:
            self._kind_box.addItem(style_setting_label(kind), kind)
        picker_row.addWidget(self._kind_box, 1)
        column.addLayout(picker_row)

        #: One page per kind, so switching kinds cannot lose an edit made on
        #: another -- which a rebuild-on-change panel would do silently.
        self._pages = QTabWidget()
        self._pages.tabBar().setVisible(False)
        #: ``{kind: {name: (getter, default)}}``.
        self._kind_controls = {}
        for kind in self._kinds:
            page = QWidget()
            form = QFormLayout(page)
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            controls = {}
            stored = self._per_graph.get(kind, {})
            for name, default in self._graph_defaults.get(kind, {}).items():
                value = stored.get(name, default)
                widget, getter, setter = self._control(name, value)
                controls[name] = (getter, setter, default)
                form.addRow(style_setting_label(name), widget)
            self._kind_controls[kind] = controls
            self._pages.addTab(page, kind)
        column.addWidget(self._pages)
        self._kind_box.currentIndexChanged.connect(self._pages.setCurrentIndex)

        # SAVE / LOAD, instruction 108 point 5, on the panel that owns these
        # settings. The same file the figure's right-click menu reads and
        # writes -- one format, two ways in, and no third place a graph style
        # can live.
        file_row = QHBoxLayout()
        save_button = QPushButton("Save style…")
        save_button.setToolTip(
            "Write these settings to a file, so a lab's house style can be "
            "shared and re-applied without setting every control again.")
        save_button.clicked.connect(self._save_to_file)
        load_button = QPushButton("Load style…")
        load_button.setToolTip(
            "Read a saved house style into these controls. Press Save to "
            "make it this project's default.")
        load_button.clicked.connect(self._load_from_file)
        file_row.addWidget(save_button)
        file_row.addWidget(load_button)
        file_row.addStretch(1)
        column.addLayout(file_row)

        # HOVER HELP BELONGS ON THE SETTING'S NAME, NOT ON THE CONTROL
        # (instruction 113, restated across every module 2026-08-19: "the
        # tooltip should only be visable when hovering the mouse over the
        # setting name text, and not when hovering over the field, checkbox,
        # or whatever the setting controlls"). One post-pass rather than a
        # convention every hand-built row has to remember -- which is what
        # `tests/test_tooltips_are_on_the_setting_not_the_field.py` exists to
        # catch, and did catch this screen.
        from ..screens.settings_model import retarget_field_tooltips
        retarget_field_tooltips(self)

    # -- a house style as a file ---------------------------------------------

    def _save_to_file(self) -> None:
        """Write WHAT IS ON SCREEN, not what is stored.

        A panel with unsaved edits that saved the store instead would write a
        file the user can see does not match the controls in front of them.
        """
        from PySide6.QtWidgets import QFileDialog

        path, _filter = QFileDialog.getSaveFileName(
            self, "Save graph style", "graph_style.json",
            "spaCR graph style (*.json);;All files (*)")
        if path:
            general, per_graph = self.values()
            save_graph_style(path, general, per_graph)

    def _load_from_file(self) -> None:
        """Read a house style INTO THE CONTROLS, not into the store.

        So the user sees what they are about to accept and can still press
        Cancel -- a load that wrote straight through would be the one action
        in this dialog that Cancel could not undo.
        """
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        path, _filter = QFileDialog.getOpenFileName(
            self, "Load graph style", "",
            "spaCR graph style (*.json);;All files (*)")
        if not path:
            return
        try:
            general, per_graph = load_graph_style(path)
        except (OSError, ValueError, JSONDecodeError) as error:
            QMessageBox.warning(self, "Load graph style", str(error))
            return
        self.apply_values(general, per_graph)

    def apply_values(self, general=None, per_graph=None) -> None:
        """Put ``general``/``per_graph`` into the controls.

        A setting the file does not mention goes back to the PACKAGE DEFAULT
        rather than keeping whatever was on screen. Loading a house style
        that leaves half of somebody else's settings behind is not the house
        style, and the deltas the file stores only mean anything against the
        defaults.
        """
        general = dict(general or {})
        per_graph = {str(kind): dict(values)
                     for kind, values in (per_graph or {}).items()
                     if isinstance(values, dict)}
        for name, (_getter, setter, default) in self._general_controls.items():
            setter(general.get(name, default))
        for kind, controls in self._kind_controls.items():
            stored = per_graph.get(kind, {})
            for name, (_getter, setter, default) in controls.items():
                setter(stored.get(name, default))

    # -- building one control ------------------------------------------------

    def _control(self, name: str, value):
        """``(widget, getter, setter)`` for one style setting, from its VALUE.

        The value and not a declared type, for the reason instruction 108
        records: a dataclass under ``from __future__ import annotations``
        carries its type as a string, and these are plain dicts with no
        annotation at all. The value is the only thing that is always there.

        THE SETTER IS BUILT HERE, beside the getter, and that is deliberate.
        "Reset to defaults" has to put every control back, and the
        alternative -- working out from the widget's class how to write to it
        -- misses exactly the newest control kind, which is the one nobody
        remembers to add to the reset. Built together, they cannot disagree.
        """
        choices = style_choices_for(name)
        if name in TRANSPARENT_CAPABLE and (_looks_like_a_colour(value)
                                            or _is_transparent_ground(value)):
            return self._ground_control(name, value)
        if isinstance(value, bool):
            box = QCheckBox()
            box.setChecked(bool(value))
            return box, box.isChecked, lambda v: box.setChecked(bool(v))
        if choices:
            combo = QComboBox()
            for option in choices:
                combo.addItem(str(option), option)
            index = combo.findData(value)
            if index < 0:
                # A stored value the package no longer offers. Kept and
                # shown rather than snapped to the first entry, because
                # silently changing a user's setting while showing them a
                # settings dialog is the worst place to do it.
                combo.addItem(f"{value} (not offered)", value)
                index = combo.count() - 1
            combo.setCurrentIndex(index)

            def _set_combo(v, box=combo):
                found = box.findData(v)
                if found >= 0:
                    box.setCurrentIndex(found)
            return combo, combo.currentData, _set_combo
        if _looks_like_a_colour(value):
            holder = {"value": str(value)}
            button = _colour_button(
                str(value), lambda chosen: holder.__setitem__("value", chosen))

            def _set_colour(v, b=button, h=holder):
                h["value"] = str(v)
                # THE SWATCH TOO. `_colour_button` paints itself from its own
                # state, so writing the holder alone would leave the button
                # showing the old colour -- a reset the user can see did not
                # happen. Rebuilt in place rather than reaching into the
                # button's private state.
                b.setText(str(v))
                colour = QColor(str(v))
                if colour.isValid():
                    ink = "#000" if colour.lightness() > 127 else "#fff"
                    b.setStyleSheet(f"background-color: {colour.name()}; "
                                    f"color: {ink};")
            return button, lambda h=holder: h["value"], _set_colour
        if isinstance(value, float):
            spin = QDoubleSpinBox()
            spin.setDecimals(2)
            spin.setRange(0.0, 1000.0)
            spin.setSingleStep(0.1)
            spin.setValue(float(value))
            return spin, spin.value, lambda v: spin.setValue(float(v))
        if isinstance(value, int):
            spin = QSpinBox()
            spin.setRange(0, 10000)
            spin.setValue(int(value))
            return spin, spin.value, lambda v: spin.setValue(int(v))
        line = QLineEdit(str(value))
        return line, line.text, lambda v: line.setText(str(v))

    def _ground_control(self, name: str, value):
        """A colour button with a "Transparent" box beside it.

        GREYED, NOT REMOVED (INVARIANTS 6): ticking Transparent disables the
        colour button rather than hiding it, so the colour the user had is
        still on screen and is still there when they untick. A control that
        vanishes takes its value with it.
        """
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        transparent = _is_transparent_ground(value)
        holder = {"value": (self._general_defaults.get(name, "#FFFFFF")
                            if transparent else str(value))}
        button = _colour_button(
            holder["value"], lambda chosen: holder.__setitem__("value", chosen))
        box = QCheckBox("Transparent")
        box.setToolTip(
            "No background at all, so the figure takes the colour of "
            "whatever it is placed on. What a figure for a manuscript "
            "usually wants -- but check the text is still legible against "
            "the page, because white text on a transparent ground is "
            "invisible in a document.")
        box.setChecked(transparent)
        layout.addWidget(button, 1)
        layout.addWidget(box)

        def _paint_button(colour: str) -> None:
            button.setText(str(colour))
            qcolour = QColor(str(colour))
            if qcolour.isValid():
                ink = "#000" if qcolour.lightness() > 127 else "#fff"
                button.setStyleSheet(f"background-color: {qcolour.name()}; "
                                     f"color: {ink};")

        def _sync(*_):
            button.setEnabled(not box.isChecked())
        box.toggled.connect(_sync)
        _sync()

        def _get():
            return (TRANSPARENT_STYLE_GROUND if box.isChecked()
                    else holder["value"])

        def _set(new_value):
            if _is_transparent_ground(new_value):
                box.setChecked(True)
            else:
                box.setChecked(False)
                holder["value"] = str(new_value)
                _paint_button(str(new_value))
            _sync()

        return row, _get, _set

    # -- reading it back -----------------------------------------------------

    def values(self) -> tuple:
        """``(general, per_graph)`` -- ONLY what differs from the defaults.

        The deltas, never the resolved style. A panel that wrote every
        control back would freeze today's defaults into every user's
        settings, so improving a default would stop reaching anybody who had
        ever opened this tab -- and, through `figures.style.user_overrides`,
        would replace the publication house style for all of them.
        """
        general = {}
        for name, (getter, _setter, default) in self._general_controls.items():
            value = getter()
            if not _same_setting(value, default):
                general[name] = value
        per_graph = {}
        for kind, controls in self._kind_controls.items():
            changed = {}
            for name, (getter, _setter, default) in controls.items():
                value = getter()
                if not _same_setting(value, default):
                    changed[name] = value
            if changed:
                per_graph[kind] = changed
        return general, per_graph

    def reset(self) -> None:
        """Put every control back to the package default.

        Preferences' "Reset to defaults" re-reads every other getter against
        a throwaway store; this panel holds its controls rather than its
        store, so it is told directly. A reset that quietly skipped this
        section would leave the graph style standing while everything around
        it moved, which reads as a broken reset.
        """
        for controls in [self._general_controls] + \
                list(self._kind_controls.values()):
            for _name, (_getter, setter, default) in controls.items():
                setter(default)

    def select_kind(self, kind: str) -> None:
        """Show one graph type's page. For a caller that knows which figure
        the user was looking at when they opened Preferences."""
        index = self._kind_box.findData(str(kind))
        if index >= 0:
            self._kind_box.setCurrentIndex(index)


def _same_setting(value, default) -> bool:
    """Whether a control still holds its default.

    Numbers are compared with a tolerance, because a QDoubleSpinBox with two
    decimals cannot hold 0.6 exactly and a panel that stored `grid_width:
    0.6000000000000001` would mark every user as having overridden a setting
    they never touched.
    """
    if isinstance(default, bool) or isinstance(value, bool):
        return bool(value) == bool(default)
    if isinstance(default, (int, float)) and isinstance(value, (int, float)):
        return abs(float(value) - float(default)) < 1e-6
    return str(value) == str(default)
