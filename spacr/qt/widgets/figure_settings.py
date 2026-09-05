"""Restyle and export live Matplotlib figures from the Qt interface.

The settings dialog builds its controls from the artists present in a figure,
so only applicable options are shown. It can update data-dependent properties,
such as axis scales, without rerunning the analysis. This module also manages
reusable graph-style files and optional data, statistics, and caption sidecars.
"""

from __future__ import annotations

import logging
import math
import os
from json import JSONDecodeError
from typing import Callable, Optional

import numpy

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
from ..i18n import tr

#: Axis scales offered by the figure settings dialog. ``symlog`` supports
#: signed values that cannot be represented on a standard logarithmic scale.
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
    except Exception:  # genuinely unreadable colour
        return fallback


def _colour_button(initial, on_pick: Callable[[str], None]) -> QPushButton:
    """A button showing a colour that opens a picker."""
    button = QPushButton()
    state = {"colour": _as_hex(initial)}

    def _paint():
        """Show the current colour on the button, as a swatch and as text."""
        colour = QColor(state["colour"])
        button.setText(state["colour"])
        if colour.isValid():
            button.setStyleSheet(
                f"background-color: {colour.name()}; "
                f"color: {'#000' if colour.lightness() > 127 else '#fff'};")

    def _choose():
        # Qt's own dialog, never the platform one -- see
        # :mod:`spacr.qt.widgets.colour_picker`.
        """Ask for a colour and keep it if the dialog returned one."""
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
    """Edit the supported appearance settings of a live figure.

    Controls are created from the figure's current axes, artists, legends, and
    optional spaCR metadata. Changes are previewed after a short debounce;
    rejecting the dialog restores the opening state when it could be captured.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure to edit.
    parent : PySide6.QtWidgets.QWidget, optional
        Parent widget.
    on_change : callable, optional
        Callback invoked after an edit. Callbacks may accept a ``preview``
        keyword argument.
    propagate_callback : callable, optional
        Callback that receives the current Image UMAP settings when the user
        selects ``Propagate settings``.
    """

    #: Debounce interval in milliseconds between an edit and its preview.
    REDRAW_DELAY_MS = 60

    def __init__(self, figure, parent=None, *, on_change: Optional[Callable] = None,
                 propagate_callback: Optional[Callable] = None):
        """Build the figure settings dialog with a live preview.

        A pickled snapshot of the figure is taken so Cancel has something to go
        back to: live apply with no way out is a trap -- the user drags a spin
        box to see what it does and there is no longer an "as it was". The
        per-figure text-size override is kept separately, because Cancel
        restores the figure by copying axes out of the snapshot rather than by
        swapping the object, so that attribute would otherwise survive an undo
        of everything it applies to.

        The Statistics tab appears only for a figure that compares groups: one
        offering a t-test on a Q-Q plot would be an invitation to report a
        number that means nothing. The UMAP tab appears only for a figure
        carrying the embedding it was drawn from -- without it, "live" would
        mean re-running the reduction and every point would move.

        :param figure: the matplotlib figure to restyle.
        :param parent: parent widget, or ``None``.
        :param on_change: called to redraw; takes ``preview`` when it can.
        :param propagate_callback: writes the values into the owning module's
            settings panel. ``None`` disables Propagate and says why.
        """
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
        except Exception:  # artists that will not pickle
            pass
        # The per-figure text size is an ATTRIBUTE, and `reject` restores the
        # figure by copying axes out of the snapshot rather than by swapping
        # the object -- so the attribute would survive a Cancel that undid
        # everything it applies to. Kept here and put back explicitly.
        try:
            from .figure_queue import figure_text_size_override
            self._text_size_at_open = figure_text_size_override(figure)
        except Exception:  # figure_queue unavailable
            self._text_size_at_open = 0

        # Coalesce redraws. Every control calls _changed(); this restarts a
        # single-shot timer, so a burst of twenty value changes costs one
        # render instead of twenty.
        #: Whether a preview render is currently running.
        self._rendering = False
        #: Whether another preview is required after the current render.
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
        except Exception:  # UMAP support absent
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
        """Return the current Image UMAP figure settings.

        Returns
        -------
        dict
            Current settings, or an empty dictionary when the figure has no
            Image UMAP controls.
        """
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
        """Restore the opening figure state and close the dialog.

        Restoration is best-effort when the figure could not be serialized or
        an artist cannot be reconstructed.
        """
        if self._snapshot is not None:
            try:
                import pickle

                restored = pickle.loads(self._snapshot)
                # Copy the restored state back INTO the figure the queue
                # holds, rather than swapping the object -- everything else
                # refers to the original by identity.
                self._figure.clear()
                for axis in list(restored.axes):
                    # DETACHED FIRST. matplotlib refuses to put one artist
                    # in two figures, and a restored axes still belongs to
                    # the figure the snapshot was unpickled into -- so
                    # re-homing it without the detach raised on the first
                    # axes and Cancel left a CLEARED figure behind, with
                    # neither the size nor the ground it opened with.
                    axis.remove()
                    self._figure._axstack.add(axis)
                    axis.set_figure(self._figure)
                self._figure.patch.set_facecolor(restored.patch.get_facecolor())
                self._figure.set_size_inches(*restored.get_size_inches())
                self._changed()
            except Exception:  # restore is best-effort
                pass
        try:
            from .figure_queue import set_figure_text_size_override
            set_figure_text_size_override(
                self._figure, getattr(self, "_text_size_at_open", 0))
        except Exception:  # figure_queue unavailable
            pass
        super().reject()

    #: Input widget types that receive wheel events only while focused.
    _WHEEL_STEALERS = (QSpinBox, QDoubleSpinBox, QComboBox)

    #: Maximum number of series that receive individual appearance controls.
    SERIES_DETAIL_LIMIT = 8

    #: Palettes offered when an axes exceeds :attr:`SERIES_DETAIL_LIMIT`.
    PALETTES = ("tab10", "tab20", "Set1", "Set2", "Set3", "Dark2", "Paired",
                "Accent", "viridis", "plasma", "cividis", "coolwarm")

    def _add_series_rules(self, form, axis, series) -> None:
        """Add shared styling controls for axes with many series.

        A single palette, size, and opacity rule applies across the complete
        series set instead of presenting one control per mark.
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
            """Recolour every series from the chosen palette."""
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
                except Exception:  # artist without colour
                    pass
            self._changed()
        palette.currentIndexChanged.connect(apply_palette)
        form.addRow("Palette", palette)

        size = QDoubleSpinBox()
        size.setRange(1.0, 600.0)
        size.setValue(36.0)

        def apply_size(value):
            """Resize every series that has a size to set."""
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
            """Set the alpha on every series."""
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
            """Set the edge width on every series that has one."""
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
        """Prevent unfocused inputs from consuming scroll-wheel events.

        Parameters
        ----------
        obj : PySide6.QtCore.QObject
            Object receiving the event.
        event : PySide6.QtCore.QEvent
            Event being filtered.

        Returns
        -------
        bool
            ``True`` when an unfocused input's wheel event was consumed;
            otherwise the result from the parent event filter.
        """
        if (event.type() == QEvent.Wheel
                and isinstance(obj, self._WHEEL_STEALERS)
                and not obj.hasFocus()):
            event.ignore()
            return True
        return super().eventFilter(obj, event)

    def closeEvent(self, event):  # noqa: N802 - Qt name
        """Complete a full-quality redraw before closing the dialog.

        Parameters
        ----------
        event : PySide6.QtGui.QCloseEvent
            Qt close event forwarded to the parent implementation.
        """
        if self._redraw.isActive():
            self._redraw.stop()
            self._redraw_now(preview=False)
        else:
            self._redraw_now(preview=False)
        super().closeEvent(event)

    # ------------------------------------------------------------- plumbing

    @staticmethod
    def _scroll(widget: QWidget) -> QScrollArea:
        """Wrap a tab page in a scroll area.

        :param widget: the page.
        :returns: the scroll area holding it.
        """
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
        """Redraw the figure, never letting renders stack.

        A preview blocks the GUI thread for about 150 ms, and Qt keeps
        delivering events during it -- spin-box auto-repeat, the wheel, this
        timer. Without the guard each one lands another render behind the
        current one, the queue grows faster than it drains, and the window stops
        responding. A request arriving mid-render only sets a flag, and one
        final redraw runs afterwards: the thread is always free between renders
        and the picture still ends up matching the controls.

        :param preview: render at preview quality rather than full.
        """
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
        """Build the statistical-test controls and show the resolved choice.

        The tab is available only for figures that carry comparable groups.
        Automatic selection is the default and reports the chosen test and
        rationale; an explicit test remains available for design information
        the data alone cannot infer.
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
            "Automatic chooses a test from the group count, Levene's test "
            "for equal variance, and Shapiro-Wilk tests for normality. If an "
            "assumption check has too few values to run, that assumption is "
            "treated as unmet.")
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
        except Exception:              # module absent
            correction.addItem("fdr_bh", "fdr_bh")
        correction.setToolTip(
            "Adjust p-values across all comparisons shown in this panel. "
            "Without correction, six independent tests at alpha 0.05 have "
            "about a 26% chance of at least one false positive.")
        form.addRow("Correct across pairs", correction)

        unit = QLineEdit("coefficient")
        unit.setToolTip(
            "Name the independent observational unit used by the test. Use "
            "wells rather than individual cells when wells are the replicates; "
            "treating correlated cells as independent can greatly overstate "
            "significance.")
        form.addRow("Unit of replication", unit)

        verdict = QLabel("")
        verdict.setWordWrap(True)
        verdict.setStyleSheet("color: palette(mid); font-size: 11px;")
        form.addRow(verdict)

        def _recompute():
            """Re-run the test with the current choices and redraw."""
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
        """Build the Figure page: size, DPI, background, and the two ink controls.

        The text-size control reaches *every* text object, including the ones a
        naive sweep misses -- annotations, the suptitle and the legend title --
        which is what made "shrink all text" leave the largest label on the
        plot untouched and read as the font getting bigger. The size is
        remembered on the figure rather than written to the preference, because
        this dialog restyles one figure in front of the user and the setting for
        every figure is Preferences.

        Line ink and font ink are two controls rather than one, split by what a
        mark is rather than by which code draws it, so "dark axes, coloured
        labels" is expressible.

        :returns: the page widget.
        """
        from .figure_queue import set_figure_text_size_override

        page = QWidget()
        form = QFormLayout(page)
        figure = self._figure

        def set_face(colour):
            """Set the figure's own background colour."""
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
            """Resize the figure to the width and height on screen."""
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
            """Set one font size on every piece of text in the figure."""
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
            """Recolour every line in the figure."""
            apply_line_colour(figure, colour)
            self._changed()

        def set_font_ink(colour):
            """Recolour every piece of text in the figure."""
            apply_font_colour(figure, colour)
            self._changed()

        current_font_colour = "#000000"
        current_line_colour = "#000000"
        if figure.axes:
            try:
                current_font_colour = _as_hex(
                    figure.axes[0].xaxis.label.get_color())
            except Exception:      # odd colour spec
                pass
            try:
                spines = list(figure.axes[0].spines.values())
                if spines:
                    current_line_colour = _as_hex(spines[0].get_edgecolor())
            except Exception:      # odd colour spec
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
        """Build one axes page: its title, labels, scales, limits and ticks.

        :param axis: the axes this page edits.
        :returns: the page widget.
        """
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
                """Apply one axis's limits, refusing a zero-width range.

                Equal bounds collapse the axis and matplotlib draws nothing, so the
                value is left alone rather than applied.
                """
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
            """Recompute the limits from the data now on the axis."""
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
            """Show or hide the grid, passing line properties ONLY when enabling.

            matplotlib warns "First parameter to grid() is false, but line
            properties are supplied" and then turns the grid ON regardless -- so
            passing them unconditionally made the checkbox unable to switch the grid
            off, which is the opposite of what it says.
            """
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
            """Set every spine's width, or hide them all at zero."""
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
            """Hide or show the top and right spines together."""
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
                """Rebuild the legend from the current choices."""
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
                """Recolour one artist. The artist is bound as a default argument.

                Bound at definition rather than closed over: a loop variable closed over
                gives every callback the LAST artist, which is the classic way a row of
                per-artist controls all end up editing one of them.
                """
                try:
                    a.set_color(colour)
                except Exception:  # artist without colour
                    pass
                self._changed()
            try:
                current = artist.get_color()
            except Exception:
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
                except Exception:
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
                except Exception:
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
            except Exception:
                alpha.setValue(1.0)
            alpha.valueChanged.connect(
                lambda value, a=artist: (a.set_alpha(value), self._changed()))
            form.addRow("  Opacity", alpha)

        return page


def figure_line_artists(figure) -> list:
    """Collect artists affected by the global line-colour control.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure whose artists should be collected.

    Returns
    -------
    list
        Data and reference lines, axes spines, and legend sample lines.

    Notes
    -----
    Gridlines are excluded. Tick marks are updated separately by
    :func:`apply_line_colour` because Matplotlib recreates them during draws.
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
    """Apply one colour to figure lines, spines, and tick marks.

    Line styles and dash patterns are preserved. Gridlines and text are not
    changed.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure to update.
    colour : Any
        Matplotlib colour specification.

    Returns
    -------
    int
        Number of line, spine, and legend artists successfully updated. Tick
        marks are updated separately and are not included in this count.
    """
    touched = 0
    for artist in figure_line_artists(figure):
        try:
            if hasattr(artist, "set_edgecolor"):
                artist.set_edgecolor(colour)     # a spine
            else:
                artist.set_color(colour)
            touched += 1
        except Exception:                        # odd spec
            continue
    for axis in getattr(figure, "axes", ()):
        # THE TICK MARKS, SEPARATELY, and `color=` only. `colors=` would set
        # the LABEL as well, which is the conflation the two controls exist
        # to undo -- and it is done through `tick_params` rather than over
        # the current ticks because matplotlib rebuilds them on every draw,
        # so a colour set on the objects is lost at the next autoscale.
        try:
            axis.tick_params(color=colour, which="both")
        except Exception:
            continue
    return touched


def apply_font_colour(figure, colour) -> int:
    """Apply one colour to every text object in a figure.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure to update.
    colour : Any
        Matplotlib colour specification.

    Returns
    -------
    int
        Number of text objects successfully updated. This includes titles,
        axes labels, tick labels, legends, and annotations.
    """
    touched = 0
    for item in _every_text(figure):
        try:
            item.set_color(colour)
            touched += 1
        except Exception:
            continue
    for axis in getattr(figure, "axes", ()):
        # The labels are regenerated on every draw, so the colour has to be
        # set on the TICK rather than only on today's label objects.
        try:
            axis.tick_params(labelcolor=colour, which="both")
        except Exception:
            continue
    return touched


def figure_follows_the_theme(figure) -> None:
    """Restore line and font colours from the active theme preferences.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure to update. If the preference store is unavailable, black is
        used for both line and font colours.
    """
    try:
        from ..preferences import get_figure_colors, get_figure_line_colour

        _bg, font = get_figure_colors()
        line = get_figure_line_colour()
    except Exception:                            # no store
        font = line = "#000000"
    apply_line_colour(figure, line)
    apply_font_colour(figure, font)


#: Schema identifier required in a spaCR graph-style JSON file.
GRAPH_STYLE_FILE_KIND = "spacr_graph_style"


def graph_style_as_dict(general=None, per_graph=None) -> dict:
    """Serialize graph-style preference overrides to a dictionary.

    Parameters
    ----------
    general : mapping, optional
        General style overrides. If ``None``, read the current preference.
    per_graph : mapping of str to mapping, optional
        Overrides keyed by graph type. If ``None``, read the current
        preference.

    Returns
    -------
    dict
        Style data with ``spacr_style_kind``, ``general``, and ``per_graph``
        keys. Per-graph values that are not mappings are omitted.

    Notes
    -----
    Only preference overrides are stored. Theme-resolved colours and package
    defaults are not captured from the currently displayed figure.
    """
    if general is None or per_graph is None:
        try:
            from ..preferences import (get_figure_style,
                                       get_figure_style_per_graph)
            if general is None:
                general = get_figure_style()
            if per_graph is None:
                per_graph = get_figure_style_per_graph()
        except Exception:                    # no store
            general, per_graph = general or {}, per_graph or {}
    return {
        "spacr_style_kind": GRAPH_STYLE_FILE_KIND,
        "general": dict(general or {}),
        "per_graph": {str(kind): dict(values)
                      for kind, values in (per_graph or {}).items()
                      if isinstance(values, dict)},
    }


def save_graph_style(path: str, general=None, per_graph=None) -> str:
    """Write graph-style preference overrides to a JSON file.

    Parameters
    ----------
    path : str
        Destination path. An empty path cancels the operation.
    general : mapping, optional
        General style overrides. If ``None``, read the current preference.
    per_graph : mapping of str to mapping, optional
        Overrides keyed by graph type. If ``None``, read the current
        preference.

    Returns
    -------
    str
        ``path`` after a successful write, or an empty string if the path is
        empty or the file cannot be written.

    Raises
    ------
    TypeError
        If a supplied setting value cannot be serialized as JSON.
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
    """Read graph-style preference overrides from a JSON file.

    Parameters
    ----------
    path : str
        Path to a graph-style file created by :func:`save_graph_style`.

    Returns
    -------
    general : dict
        General style overrides.
    per_graph : dict
        Style overrides keyed by graph type.

    Raises
    ------
    OSError
        If the file cannot be opened.
    json.JSONDecodeError
        If the file does not contain valid JSON.
    ValueError
        If the file is not identified by :data:`GRAPH_STYLE_FILE_KIND`.

    Notes
    -----
    Unknown setting names are preserved for compatibility with files created
    by other spaCR versions.
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
    """Save graph-style overrides as the active preferences.

    Parameters
    ----------
    general : mapping
        General style overrides.
    per_graph : mapping of str to mapping
        Style overrides keyed by graph type. Values that are not mappings are
        omitted.
    """
    from ..preferences import set_figure_style, set_figure_style_per_graph

    set_figure_style(dict(general or {}))
    set_figure_style_per_graph({str(kind): dict(values)
                                for kind, values in (per_graph or {}).items()
                                if isinstance(values, dict)})


def add_graph_style_file_entries(menu, parent=None, *, on_change=None) -> None:
    """Add graph-style save and load actions to a menu.

    Parameters
    ----------
    menu : PySide6.QtWidgets.QMenu
        Menu that receives the actions.
    parent : PySide6.QtWidgets.QWidget, optional
        Parent for file dialogs and actions. If ``None``, use ``menu``.
    on_change : callable, optional
        Callback invoked after a style is loaded. Callbacks may accept a
        ``preview`` keyword argument.
    """
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    owner = parent if parent is not None else menu

    def _save():
        """Write the current graph style to a JSON file."""
        path, _filter = QFileDialog.getSaveFileName(
            owner, "Save graph style", "graph_style.json",
            "spaCR graph style (*.json);;All files (*)")
        if path:
            save_graph_style(path)

    def _load():
        """Read a graph style back and apply it."""
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

    save = QAction(tr("Save graph style…"), owner)
    save.setToolTip(tr(
        "Write the general and per-graph settings from Preferences to a "
        "file, so a lab's house style can be shared and re-applied."))
    save.triggered.connect(_save)
    menu.addAction(save)

    load = QAction(tr("Load graph style…"), owner)
    load.setToolTip(tr(
        "Read a saved house style and make it this project's default, so "
        "every figure drawn from now on uses it."))
    load.triggered.connect(_load)
    menu.addAction(load)


#: Grouped-plot types offered by the figure context menu, in display order.
GROUPED_PLOT_TYPES = (
    ("line", "Line"),
    ("bar", "Bar"),
    ("jitter_bar", "Jitter over bar"),
    ("jitter_box", "Jitter over box"),
    ("jitter", "Jitter"),
    ("box", "Box"),
    ("violin", "Violin"),
)


#: Column names the derived frame uses.
#:
#: A figure that was not drawn by `create_grouped_plot` has no recipe and so
#: no column names either. These are what the reconstructed frame calls its
#: two columns, and they are what the axis labels are replaced by when the
#: figure is redrawn -- so they are read by a user, not only by the drawer.
DERIVED_GROUP = "group"
DERIVED_VALUE = "value"


def _tick_labels(axes) -> dict:
    """Map x position -> tick text, for an axis with categorical ticks."""
    out = {}
    try:
        locations = list(axes.get_xticks())
        texts = [t.get_text() for t in axes.get_xticklabels()]
    except Exception:                                        # noqa: BLE001
        return out
    for position, text in zip(locations, texts):
        text = str(text).strip()
        if text:
            out[round(float(position), 6)] = text
    return out


def _named(labels, x):
    """The tick label at ``x``, or the number itself when there is none."""
    key = round(float(x), 6)
    if key in labels:
        return labels[key]
    nearest = min(labels, key=lambda k: abs(k - key)) if labels else None
    if nearest is not None and abs(nearest - key) < 0.5:
        return labels[nearest]
    return f"{float(x):g}"


def _pairs_from_axes(axes):
    """Every (group, value) pair an axes actually drew, or an empty list.

    Reads the ARTISTS rather than any data the caller kept, because for these
    figures nobody kept any: this is the path for a figure that arrived
    without a recipe. Three artist families cover what spaCR draws --
    rectangles for bars, path collections for scatter and strip, and lines
    for series and for the markers matplotlib draws as lines.
    """
    labels = _tick_labels(axes)
    pairs = []

    # BARS. Height is the value and the bar's centre picks the tick label.
    # Patches that span the whole axes are backgrounds, not data.
    try:
        from matplotlib.patches import Rectangle

        span = axes.get_xlim()
        width = abs(span[1] - span[0]) or 1.0
        for patch in list(axes.patches):
            if not isinstance(patch, Rectangle):
                continue
            if patch.get_width() >= width * 0.98:
                continue
            height = patch.get_height()
            if height is None or not math.isfinite(float(height)):
                continue
            centre = patch.get_x() + patch.get_width() / 2.0
            pairs.append((_named(labels, centre), float(height)))
    except Exception:                                        # noqa: BLE001
        pass

    # SCATTER AND STRIP.
    try:
        for collection in list(axes.collections):
            offsets = collection.get_offsets()
            if offsets is None or len(offsets) == 0:
                continue
            for x, y in numpy.asarray(offsets, dtype=float):
                if math.isfinite(x) and math.isfinite(y):
                    pairs.append((_named(labels, x), float(y)))
    except Exception:                                        # noqa: BLE001
        pass

    # LINES AND MARKERS. A line with no marker and two points is usually a
    # reference line rather than data, and is left out.
    try:
        for line in list(axes.lines):
            data = line.get_xydata()
            if data is None or len(data) == 0:
                continue
            if len(data) <= 2 and (line.get_marker() in (None, "", "None")):
                continue
            for x, y in numpy.asarray(data, dtype=float):
                if math.isfinite(x) and math.isfinite(y):
                    pairs.append((_named(labels, x), float(y)))
    except Exception:                                        # noqa: BLE001
        pass

    return pairs


def derive_replot_recipe(figure):
    """Derive a grouped-plot recipe from an existing matplotlib figure.

    This fallback supports figures without a ``_spacr_replot`` payload by
    reading plotted values from bars, scatter collections, and data lines on a
    single axes. Artist-derived data are exact for bar heights and point
    coordinates but are necessarily lossy for summary artists such as box or
    violin plots, which do not retain their source observations. Figures
    created by :func:`spacr.plot.create_grouped_plot` use their attached source
    frame instead of this fallback.

    :param figure: Matplotlib ``Figure`` containing exactly one axes.
    :returns: Recipe dictionary for :func:`spacr.plot.create_grouped_plot`, or
        ``None`` when sufficient plottable values cannot be recovered.
    """
    try:
        import pandas
    except Exception:                                        # noqa: BLE001
        return None
    axes = [a for a in getattr(figure, "axes", []) if a is not None]
    if len(axes) != 1:
        # ONE AXES ONLY. A grid of panels redrawn as a single violin would
        # throw away every panel but one, silently.
        return None
    pairs = _pairs_from_axes(axes[0])
    if len(pairs) < 2:
        return None
    frame = pandas.DataFrame(pairs, columns=[DERIVED_GROUP, DERIVED_VALUE])
    # NO `nunique() < 1` GUARD. The group column comes only from
    # `_named`, which returns either a tick label or `f"{float(x):g}"` --
    # always a str, never a missing value, and "nan" for a NaN x rather
    # than NaN itself. With at least two rows guaranteed above,
    # `nunique()` is 1 or more by construction.
    #
    # It could only be reached by making `_pairs_from_axes` return actual
    # NaN group names, which is not a figure. Instruction 310 A15 counted
    # it, and a reader maintaining this was being told nameless groups
    # are a case that occurs.
    return {
        "df": frame,
        "grouping_column": DERIVED_GROUP,
        "data_column": DERIVED_VALUE,
        "graph_type": "",
        "summary_func": "mean",
        "order": None,
        "colors": None,
        "y_lim": None,
        "error_bar_type": "std",
    }


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


def _which_types_fit(recipe) -> tuple:
    """``(fitting kinds, {kind: why not})`` for this figure's data.

    THROUGH `spacr.graph_types`, which is where the fitness table lives --
    a second opinion here would let the menu offer a type the drawer cannot
    draw, which is the failure that table exists to prevent.

    An empty first element means "could not tell", and then EVERY type is
    offered: refusing them all because the shape could not be read would
    take a working menu away over a question nobody asked.
    """
    try:
        from ...graph_types import offer, shape_of

        frame = recipe.get("df")
        if frame is None or not len(frame):
            return (), {}
        shape = shape_of(frame, str(recipe.get("grouping_column") or ""),
                         str(recipe.get("data_column") or ""))
        rows = offer(frame, str(recipe.get("grouping_column") or ""),
                     str(recipe.get("data_column") or ""))
        del shape
        fits, why = [], {}
        for kind, _caption, reason in rows:
            (why.__setitem__(kind, reason) if reason
             else fits.append(kind))
        # The two vocabularies differ: `graph_types` says `bar_jitter` where
        # the drawer says `jitter_bar`, and it has no `jitter_box`. Map the
        # ones that mean the same thing rather than renaming either -- one
        # is the analysis vocabulary and the other the drawer's, and each is
        # right in its own module.
        alias = {"bar_jitter": ("jitter_bar", "jitter_box")}
        for source, targets in alias.items():
            if source in fits:
                fits.extend(targets)
            elif source in why:
                for target in targets:
                    why.setdefault(target, why[source])
        return tuple(fits), why
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not work out which graph types fit", exc_info=True)
        return (), {}


def _add_group_colours(menu, figure, recipe, on_change, parent) -> None:
    """Add persistent colour controls for the groups represented in a plot.

    Colours are stored on the redraw recipe so all marks in a group retain
    the selected colour when the graph is redrawn or retyped.
    """
    from PySide6.QtWidgets import QMenu

    frame = recipe.get("df")
    column = str(recipe.get("grouping_column") or "")
    if frame is None or column not in getattr(frame, "columns", ()):
        return
    try:
        groups = [str(g) for g in frame[column].astype(str).unique()]
    except Exception:                                        # noqa: BLE001
        return
    if not groups:
        return

    colours = QMenu(tr("Group colours"), menu)
    colours.setToolTipsVisible(True)
    menu.addMenu(colours)

    def _recolour(group: str) -> None:
        """Pick a colour for one group and store it on the recipe."""
        current = dict(recipe.get("colors") or {})
        start = str(current.get(group, "#4C72B0"))
        chosen = pick_colour(parent, start, tr("Colour for {group}",
                                               group=group))
        if not chosen.isValid():
            return
        current[group] = chosen.name()
        # STORED ON THE RECIPE AND REDRAWN, not painted onto the artists.
        # Setting an artist's colour lasts until the next redraw and then
        # silently reverts -- which is what "changing the colors changes
        # nothing" looks like from the other side.
        recipe["colors"] = current
        figure._spacr_replot = recipe
        _replot(figure, str(recipe.get("graph_type") or "bar"), on_change)

    for group in groups[:24]:
        action = colours.addAction(f"{group}…")
        action.setToolTip(
            tr("Colour every mark belonging to {group}.", group=group))
        action.triggered.connect(
            lambda _checked=False, g=group: _recolour(g))
    if len(groups) > 24:
        # NAMED, NOT SILENTLY DROPPED. A menu that shows the first
        # twenty-four of ninety groups and says nothing looks like a menu
        # that has them all.
        note = colours.addAction(
            tr("({count} more groups not listed)", count=len(groups) - 24))
        note.setEnabled(False)


def _add_bundle_save(menu, figure, parent) -> None:
    """Add a Matplotlib action that exports the figure and its evidence bundle."""
    from PySide6.QtWidgets import QFileDialog

    action = menu.addAction(tr("Save"))
    action.setToolTip(tr(
        "Writes a FOLDER: the figure as pdf and png, the rows it was drawn "
        "from, and the test that was run on them with its assumptions. A pdf "
        "on its own cannot be checked -- six months later the question is "
        "what the numbers were and whether the difference was tested, and a "
        "figure file answers neither."))

    def _save() -> None:
        """Ask for a folder and write the whole bundle into it."""
        folder = QFileDialog.getExistingDirectory(
            parent, "Save the graph, its data and its statistics")
        if not folder:
            return
        try:
            save_figure_bundle(figure, folder)
        except Exception:                                    # noqa: BLE001
            LOG.debug("could not write the figure bundle", exc_info=True)

    action.triggered.connect(_save)


def save_figure_bundle(figure, folder: str, name: str = "") -> str:
    """Export a Matplotlib figure with its source data and statistics.

    Group definitions come from the attached replot recipe so statistical
    comparisons match the displayed figure. When no recipe is available, the
    standard files are still written and the statistics artifact records that
    no comparison could be formed.

    :param figure: Matplotlib figure to export.
    :param folder: destination directory for the bundle.
    :param name: optional base name for generated files.
    :returns: path to the written bundle directory.
    """
    from ...figures.bundle import save

    recipe = dict(getattr(figure, "_spacr_replot", None) or {})
    frame = recipe.get("df")
    column = str(recipe.get("grouping_column") or "")
    value = str(recipe.get("data_column") or "")
    groups = None
    if frame is not None and column in getattr(frame, "columns", ()) \
            and value in getattr(frame, "columns", ()):
        # `observed=True`: a categorical grouping column would otherwise
        # yield a group per unused CATEGORY as well, and an empty group is a
        # comparison arm with no observations in it.
        groups = {str(key): part[value].dropna().to_numpy()
                  for key, part in frame.groupby(column, observed=True)}

    title = name or _figure_title(figure) or "graph"

    def _render(path: str) -> None:
        # A bundle deliberately contains both formats, but each rendering
        # still uses the shared export path so print colours, embedded fonts,
        # and raster DPI match every other figure the user keeps.
        """Render one file of the bundle through the SHARED export path.

        Both formats go through `save_figure` rather than each drawing itself, so
        print colours, embedded fonts and raster DPI match every other figure the
        user keeps.
        """
        from ...plot import save_figure

        extension = os.path.splitext(path)[1].lower().lstrip(".")
        save_figure(figure, path, fmt=extension, bbox_inches="tight",
                    close=False)

    return save(folder, title, render=_render, data=frame, groups=groups,
                unit=str(recipe.get("unit") or "observation"),
                settings={k: v for k, v in recipe.items() if k != "df"})


def _figure_title(figure) -> str:
    """The figure's own title, for naming its folder."""
    try:
        if figure._suptitle is not None:
            return str(figure._suptitle.get_text())
    except Exception:                                        # noqa: BLE001
        pass
    for axis in getattr(figure, "axes", ()):
        text = str(axis.get_title() or "")
        if text:
            return text
    return ""


def build_figure_context_menu(parent, figure, *, on_change=None,
                              open_settings=None) -> QMenu:
    """Build the context menu for a displayed figure.

    The menu provides direct legend, grid, scale, colour, and export actions.
    Figures with grouped-plot metadata can also be redrawn as another plot
    type. More detailed controls are delegated to ``open_settings``.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget
        Parent for the returned menu and its dialogs.
    figure : matplotlib.figure.Figure or None
        Figure to edit. If ``None``, the menu contains a disabled status
        action.
    on_change : callable, optional
        Callback invoked after a direct edit. Callbacks may accept a
        ``preview`` keyword argument or a replacement figure.
    open_settings : callable, optional
        Callback invoked by the ``Figure settings`` action.

    Returns
    -------
    PySide6.QtWidgets.QMenu
        Context menu owned by ``parent``.
    """
    menu = QMenu(parent)
    # AN OWNER FOR THE ACTIONS, WHICH IS NOT ALWAYS `parent`. `QMenu.addAction`
    # does not adopt an action built here, so a QAction whose only reference is
    # a local name and whose parent is `None` is collected the moment this
    # function returns -- and the menu comes back holding Save, the two
    # submenus and nothing else. `add_graph_style_file_entries` already falls
    # back this way for the same reason.
    owner = parent if parent is not None else menu
    if figure is None:
        action = QAction(tr("This figure can no longer be restyled"), owner)
        action.setEnabled(False)
        menu.addAction(action)
        return menu

    axes = list(figure.axes)

    # SHOW THE SAME DATA ANOTHER WAY (178 A). Offered only where the figure
    # carries its own recipe -- `create_grouped_plot` attaches one -- because
    # a menu entry that cannot redraw the figure it is on is worse than an
    # absent one. Every other figure in spaCR simply does not get the group.
    recipe = getattr(figure, "_spacr_replot", None)
    if not (isinstance(recipe, dict) and recipe.get("df") is not None):
        # NO RECIPE, SO READ ONE BACK OFF THE AXES. Only
        # `create_grouped_plot` attaches `_spacr_replot`, which left the
        # menu on a handful of figures and absent from every other plot in
        # the software. Derived recipes are marked so a redraw does not
        # claim to be the original data.
        derived = derive_replot_recipe(figure)
        if derived is not None:
            recipe = derived
            try:
                figure._spacr_replot = derived
                figure._spacr_replot_derived = True
            except Exception:                                # noqa: BLE001
                pass
    if isinstance(recipe, dict) and recipe.get("df") is not None:
        # Give Python and C++ an explicit ownership chain. ``addMenu(str)``
        # can leave the Python wrapper as the submenu's only live owner, so a
        # caller retrieving it through the parent action gets an already
        # deleted QMenu. This is the same lifetime rule used by Appearance
        # and Axis scale below.
        # NAMED "Graph type", which is what it was asked for by: "an option
        # when i right click on a graph, called graph type that would allow
        # the user to switch between graph types".
        show_as = QMenu(tr("Graph type"), menu)
        menu.addMenu(show_as)
        current = str(recipe.get("graph_type") or "")
        # ONLY THE TYPES THAT FIT THE DATA (instruction 200 A), and the rest
        # greyed with the reason rather than absent -- a list that silently
        # shortens leaves the user wondering whether they misremembered.
        fits, why_not = _which_types_fit(recipe)
        for kind, label in GROUPED_PLOT_TYPES:
            action = show_as.addAction(label)
            action.setCheckable(True)
            action.setChecked(kind == current)
            reason = why_not.get(kind, "")
            if fits and kind not in fits:
                action.setEnabled(False)
                action.setToolTip(reason)
            else:
                action.triggered.connect(
                    lambda _checked=False, k=kind: _replot(figure, k,
                                                           on_change))
        show_as.setToolTipsVisible(True)

        # THE GROUPS, NOT THE ELEMENTS (reported 2026-08-21: "i also want to
        # modify thing on the group level not individual points and barts").
        # The Appearance menu below colours the FURNITURE -- spines, ticks,
        # text -- which is why changing a colour there appeared to do
        # nothing to the bars: it was never about them.
        _add_group_colours(menu, figure, recipe, on_change, parent)

    # AND THE WHOLE THING, on every figure that has its data (instruction
    # 223). This was on the pyqtgraph plots only, which is not where these
    # graphs are drawn -- so a feature that existed was unreachable from
    # where the user was looking.
    _add_bundle_save(menu, figure, parent)

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
        """Run ``func`` against every axis in the figure."""
        for axis in axes:
            func(axis)
        _notify()

    legend_present = any(a.get_legend() is not None for a in axes)
    legend_action = QAction(tr("Legend"), owner)
    legend_action.setCheckable(True)
    legend_action.setChecked(
        legend_present and all(a.get_legend().get_visible()
                               for a in axes if a.get_legend() is not None))

    def toggle_legend(checked):
        """Show or hide the legend on every axis."""
        for axis in axes:
            existing = axis.get_legend()
            if existing is not None:
                existing.set_visible(checked)
            elif checked and axis.get_legend_handles_labels()[0]:
                axis.legend()
        _notify()
    legend_action.toggled.connect(toggle_legend)
    menu.addAction(legend_action)

    grid_action = QAction(tr("Grid"), owner)
    grid_action.setCheckable(True)
    grid_action.setChecked(any(line.get_visible()
                               for axis in axes
                               for line in axis.get_xgridlines()))
    grid_action.toggled.connect(
        lambda checked: _apply(lambda a: a.grid(checked)))
    menu.addAction(grid_action)

    scales = QMenu(tr("Axis scale"), menu)  # see "Appearance" below for why
    menu.addMenu(scales)
    for name, setter in (("X", "set_xscale"), ("Y", "set_yscale")):
        submenu = QMenu(name, scales)
        scales.addMenu(submenu)
        for scale in AXIS_SCALES:
            action = QAction(tr(scale), owner)
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
    appearance = QMenu(tr("Appearance"), menu)
    menu.addMenu(appearance)

    def _pick_ink(title, apply_to):
        """Pick a colour and apply it through ``apply_to``."""
        current = "#000000"
        try:
            if axes:
                current = _as_hex(axes[0].xaxis.label.get_color())
        except Exception:
            pass
        chosen = pick_colour(parent, current, title)
        if chosen.isValid():
            apply_to(figure, chosen.name())
            _notify()

    line_action = QAction(tr("Line colour…"), owner)
    line_action.setToolTip(tr(
        "Every line in the figure, the axis spines and the tick marks "
        "included. The numbers beside the ticks are text and follow the "
        "font colour."))
    line_action.triggered.connect(
        lambda: _pick_ink(tr("Line colour"), apply_line_colour))
    appearance.addAction(line_action)

    font_action = QAction(tr("Font colour…"), owner)
    font_action.setToolTip(tr(
        "Every piece of text in the figure: the title, the axis labels, the "
        "tick labels, the legend and any annotation."))
    font_action.triggered.connect(
        lambda: _pick_ink(tr("Font colour"), apply_font_colour))
    appearance.addAction(font_action)

    theme_action = QAction(tr("Follow the theme (colours)"), owner)
    theme_action.setToolTip(tr(
        "Put both colours back to the app theme and the figure preferences."))
    theme_action.triggered.connect(
        lambda: (figure_follows_the_theme(figure), _notify()))
    appearance.addAction(theme_action)

    menu.addSeparator()
    save = QAction(tr("Save figure as…"), owner)
    save.setToolTip(tr(
        "Write this figure to a file using its current plot styling and the "
        "configured export background, format and resolution."))
    save.triggered.connect(lambda: save_figure_as(parent, figure))
    menu.addAction(save)

    # STYLE IT FOR THE FILE FIRST (178 C.2). "the user should be able to
    # change all of theis for the saved graph, get a preview then save."
    # Beside the direct save rather than replacing it: writing what is on
    # screen is one click and remains one click.
    styled = QAction(tr("Save figure with a preview…"), owner)
    styled.setToolTip(tr(
        "Choose the ink, background, grid, size and resolution for the saved "
        "file, preview the result, then export it. The figure on screen is "
        "not changed."))
    styled.triggered.connect(lambda: _open_styled_save(parent, figure))
    menu.addAction(styled)

    add_graph_style_file_entries(menu, parent, on_change=on_change)

    settings = QAction(tr("Figure settings…"), owner)
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
    """Save a figure using its current styling and export preferences.

    Parameters
    ----------
    parent : PySide6.QtWidgets.QWidget or None
        Parent for the file chooser when ``path`` is empty.
    figure : matplotlib.figure.Figure or None
        Figure to save. ``None`` cancels the operation.
    path : str, optional
        Destination path. If empty, prompt for a PNG, PDF, or SVG path.

    Returns
    -------
    str
        Path returned by the writer, or an empty string when saving is
        cancelled or fails.

    Notes
    -----
    A recognized filename extension takes precedence over the default output
    format. Available data, statistics, and caption metadata are exported by
    :func:`export_sidecars` beside the requested path.
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
    except Exception:                    # Qt-only build
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
    except Exception:                    # no settings store
        background, dpi = "none", 200

        def figure_bg_is_transparent(value):
            """Whether a stored ground value means "no background at all"."""
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
    """Export available figure data, statistics, and caption sidecars.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure carrying optional ``_spacr_data``, ``_spacr_groups``, or
        ``_spacr_caption`` metadata.
    path : path-like
        Figure output path. Sidecars use the same directory and basename.

    Returns
    -------
    list of str
        Successfully written sidecar paths. Depending on available metadata,
        these may include ``<name>.csv``, ``<name>_stats.csv``, and
        ``<name>_legend.txt``.

    Notes
    -----
    The data sidecar contains the rows attached to the rendered figure. The
    statistics sidecar contains all usable pairwise comparisons with multiple
    testing correction provided by spaCR's statistics table helper. Individual
    sidecar failures are logged and do not interrupt the remaining exports.
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

# Fallback choices keep the preferences panel usable if ``figure_style``
# cannot be imported. Normal operation reads the canonical choices from that
# module through ``style_choices_for``.
_FALLBACK_CHOICES = {
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
    "spines": ("all", "left_bottom", "none"),
}

#: Compatibility alias for the fallback style-choice mapping.
STYLE_CHOICES = _FALLBACK_CHOICES


def style_choices_for(name: str) -> tuple:
    """Return the choices available for a style setting.

    Parameters
    ----------
    name : str
        Style setting name.

    Returns
    -------
    tuple
        Canonical choices from :mod:`spacr.figure_style`. If that module is
        unavailable, return the local fallback choices. An empty tuple denotes
        a free-form or unknown setting.
    """
    try:
        from ...figure_style import style_choices

        return tuple(style_choices(name))
    except Exception:                   # import guard
        return tuple(_FALLBACK_CHOICES.get(name, ()))


#: Matplotlib colour value used to store a transparent figure background.
TRANSPARENT_STYLE_GROUND = "none"

#: Style keys that support an explicit transparent value in the preferences.
TRANSPARENT_CAPABLE = ("background",)


def _looks_like_a_colour(value) -> bool:
    return isinstance(value, str) and value.startswith("#")


def _is_transparent_ground(value) -> bool:
    """Whether a stored style value means "no ground at all"."""
    return str(value).strip().lower() in ("none", "transparent", "")


#: Style keys whose capitalised name is not what the setting is called.
#:
#: `aspect` is the case this exists for. Capitalised it reads "Aspect",
#: which a reader takes for the aspect RATIO -- a number tying one y unit
#: to n x units, which is a statement about the data and is a different
#: setting living under Axes. This one offers "equal" and "auto", which is
#: matplotlib's axes aspect: whether one y unit is drawn the same length as
#: one x unit. That is a statement about the DATA, not about the panel's
#: proportions -- those are the separate Page shape row -- so it is called
#: what the graph's own right-click menu calls the same control. Labelling
#: it "Graph shape" left two shape-sounding rows, no axis-lock row, and a
#: row whose caption and whose own explanation disagreed.
_STYLE_LABELS = {
    "aspect": "Lock axis scales",
}


def style_setting_label(name: str) -> str:
    """Convert a style setting name to a display label.

    Parameters
    ----------
    name : str
        Underscore-delimited style key, such as ``'grid_colour'``.

    Returns
    -------
    str
        Capitalized, space-delimited label, such as ``'Grid colour'``.
    """
    return _STYLE_LABELS.get(
        str(name), str(name).replace("_", " ").strip().capitalize())


class FigureStylePreferences(QWidget):
    """Edit general and graph-specific figure-style preferences.

    General settings apply to every figure. Each graph type can override only
    the settings it needs, and the panel stores differences from package
    defaults rather than a fully resolved style.

    Parameters
    ----------
    general : mapping, optional
        Stored general style overrides.
    per_graph : mapping of str to mapping, optional
        Stored style overrides keyed by graph type.
    parent : PySide6.QtWidgets.QWidget, optional
        Parent widget.
    """

    def __init__(self, general=None, per_graph=None, parent=None):
        """Build the figure-style preference page.

        :param general: the saved general style values; missing keys fall back
            to the shipped defaults.
        :param per_graph: saved per-graph-kind overrides, keyed by kind.
        :param parent: parent widget, or ``None``.
        """
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
        #: General controls mapped to their getter, setter, and default value.
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

        #: Persistent control page for each graph type.
        self._pages = QTabWidget()
        self._pages.tabBar().setVisible(False)
        #: Per-graph controls mapped to their getter, setter, and default value.
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
        """Load style overrides into the preference controls.

        Parameters
        ----------
        general : mapping, optional
            General style overrides.
        per_graph : mapping of str to mapping, optional
            Style overrides keyed by graph type.

        Notes
        -----
        A control omitted from the supplied mappings is reset to its package
        default rather than retaining its previous value.
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
        """Build a widget, getter, and setter from one style value.

        Runtime values determine the control type because annotations may be
        strings or absent. Returning the setter beside the getter ensures
        reset-to-default behavior uses the same control contract.
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
                """Select the entry whose data is ``v``, if the box has one."""
                found = box.findData(v)
                if found >= 0:
                    box.setCurrentIndex(found)
            return combo, combo.currentData, _set_combo
        if _looks_like_a_colour(value):
            holder = {"value": str(value)}
            button = _colour_button(
                str(value), lambda chosen: holder.__setitem__("value", chosen))

            def _set_colour(v, b=button, h=holder):
                """Store a colour and repaint the swatch that shows it."""
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
            "Remove the figure and axes background so the underlying page or "
            "slide shows through. Check text, axes and data colours against "
            "the destination background before exporting.")
        box.setChecked(transparent)
        layout.addWidget(button, 1)
        layout.addWidget(box)

        def _paint_button(colour: str) -> None:
            """Show the ground colour on the button, as swatch and text."""
            button.setText(str(colour))
            qcolour = QColor(str(colour))
            if qcolour.isValid():
                ink = "#000" if qcolour.lightness() > 127 else "#fff"
                button.setStyleSheet(f"background-color: {qcolour.name()}; "
                                     f"color: {ink};")

        def _sync(*_):
            """Grey the colour button while transparent is ticked."""
            button.setEnabled(not box.isChecked())
        box.toggled.connect(_sync)
        _sync()

        def _get():
            """The ground: the transparent sentinel, or the chosen colour."""
            return (TRANSPARENT_STYLE_GROUND if box.isChecked()
                    else holder["value"])

        def _set(new_value):
            """Apply a ground, ticking transparent when that is what it means."""
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
        """Return style settings that differ from package defaults.

        Returns
        -------
        general : dict
            General style overrides.
        per_graph : dict
            Non-default style settings keyed by graph type. Graph types with
            no overrides are omitted.
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
        """Reset every style control to its package default."""
        for controls in [self._general_controls] + \
                list(self._kind_controls.values()):
            for _name, (_getter, setter, default) in controls.items():
                setter(default)

    def select_kind(self, kind: str) -> None:
        """Display the preference page for one graph type.

        Parameters
        ----------
        kind : str
            Graph type from :data:`spacr.figure_style.GRAPH_KINDS`. Unknown
            values leave the current page unchanged.
        """
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
