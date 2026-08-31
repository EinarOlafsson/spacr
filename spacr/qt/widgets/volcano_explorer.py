"""The volcano you can click, restyle and export.

The pipeline writes a volcano PDF, and until now that was the end of it: to
change an axis label, recolour by a covariate or find out which guide a dot
was, you re-ran the analysis or opened the CSV. This widget makes the plot the
thing you interrogate.

Three ideas hold it together:

* **One renderer.** Every pixel comes from
  :func:`spacr.volcano_style.render_volcano`, the same function the headless
  pipeline calls. Exporting re-renders from the same :class:`VolcanoStyle` at
  print size rather than screenshotting the widget, so what is saved is what
  was on screen, at full vector quality.
* **The style is a value.** Controls write into a
  :class:`~spacr.volcano_style.VolcanoStyle` and ask for a redraw. That makes
  the whole appearance saveable to JSON, reloadable, and reproducible.
* **Clicking is a lookup, not a hit test on pixels.** The nearest point is
  found in data space normalised by the axis ranges, so a click is as accurate
  on a squashed axis as on a square one.

Annotation files can be dropped in to colour or shape the points by anything:
the file is merged onto the results by a join column, and every column it
brings becomes available in the colour-by and shape-by menus.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...volcano_style import (
    COLORMAPS,
    FONT_FAMILIES,
    LINE_STYLES,
    MARKER_SHAPES,
    SCALES,
    VolcanoStyle,
    localizations_present,
    point_details,
    render_volcano,
    validate_style,
)
from .sortable_table import install_sorting, table_item

#: What an optional column menu calls "leave this unset". Spelled once,
#: because the menu has to tell it apart from a column actually named that.
_NONE_ROW = "— none —"

#: The red a broken setting's name turns, when the theme cannot be asked.
#: spaCR's own error hue; a literal only for a bare widget with no palette
#: behind it, which is a unit run rather than the application.
_PROBLEM_INK = "#C4441C"


def _setting_named_in(message: str) -> str:
    """The style setting a renderer error blames, or ``""`` for none.

    The renderer says ``x_column='foo' is not a column of the results``, so
    the setting is usually spelled out in the sentence -- and where it is,
    the reader can be pointed at the control that caused it rather than left
    to hunt. Longest name first, or ``color_by`` would answer for
    ``color_by`` and ``colormap`` alike; the ``name=`` form is preferred over
    a bare mention, because a message may name a second setting only to
    suggest it.

    ``""`` when nothing in the sentence is a setting: the explanation is
    still shown, it just has no control to turn red.
    """
    text = str(message)
    names = sorted((f.name for f in dataclasses.fields(VolcanoStyle)),
                   key=len, reverse=True)
    for name in names:
        if f"{name}=" in text:
            return name
    for name in names:
        if name in text:
            return name
    return ""


#: Columns never offered as a colour/shape source: they are the plot's own
#: axes or bookkeeping, and mapping colour to the y axis says nothing.
_NON_MAPPING_COLUMNS = frozenset({
    "standardized_marginal_effect", "adjusted_p_value", "permutation_p_value",
    "p_value", "q_value", "coefficient", "significant", "alpha",
})


class _OptionalNumbers(QWidget):
    """One number, or a pair of them, that can also be "automatic".

    An axis limit and the ends of a colour scale are ``None`` for "let the
    data decide" and a number otherwise, and a spin box has no way to say
    ``None``: a panel built from spin boxes alone can set a limit and never
    take it back off. The tick is that third state.
    """

    changed = Signal()

    def __init__(self, count: int, low: float, high: float, decimals: int,
                 caption: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._auto = QCheckBox("auto", self)
        self._auto.setToolTip("Let the plot choose. Untick to type a value.")
        self._auto.setChecked(True)
        layout.addWidget(self._auto)
        self._spins: list[QDoubleSpinBox] = []
        for index in range(int(count)):
            spin = QDoubleSpinBox(self)
            spin.setRange(low, high)
            spin.setDecimals(decimals)
            spin.setToolTip(f"{caption} ({'from' if index == 0 else 'to'})"
                            if count > 1 else caption)
            layout.addWidget(spin)
            self._spins.append(spin)
        self._sync_enabled()
        # Connected LAST, so building the widget is not a change to report.
        self._auto.toggled.connect(self._auto_toggled)
        for spin in self._spins:
            spin.valueChanged.connect(lambda _value: self.changed.emit())

    def _auto_toggled(self, _on: bool) -> None:
        self._sync_enabled()
        self.changed.emit()

    def _sync_enabled(self) -> None:
        for spin in self._spins:
            spin.setEnabled(not self._auto.isChecked())

    def value(self):
        """``None`` when automatic, else the number or the pair."""
        if self._auto.isChecked():
            return None
        numbers = tuple(float(spin.value()) for spin in self._spins)
        return numbers[0] if len(numbers) == 1 else numbers

    def setValue(self, value) -> None:  # noqa: N802 - matches QSpinBox
        widgets = [self._auto, *self._spins]
        blocked = [widget.blockSignals(True) for widget in widgets]
        try:
            numbers = []
            if value is not None:
                given = ([value] if len(self._spins) == 1 else list(value))
                try:
                    numbers = [float(item) for item in given]
                except (TypeError, ValueError):
                    numbers = []
            if len(numbers) == len(self._spins):
                self._auto.setChecked(False)
                for spin, number in zip(self._spins, numbers):
                    spin.setValue(number)
            else:
                # A value this control cannot express -- a pair where one
                # number was wanted -- reads as automatic rather than as a
                # number nobody typed.
                self._auto.setChecked(True)
        finally:
            for widget, was in zip(widgets, blocked):
                widget.blockSignals(was)
        self._sync_enabled()


class _MultiSelect(QListWidget):
    """A closed list any number of whose entries can be ticked at once.

    Several at once because "dense granules and rhoptries 1" is ONE question:
    a combo box would make the reader choose which half of their comparison
    to look at.
    """

    changed = Signal()

    def __init__(self, caption: str, parent=None):
        super().__init__(parent)
        self.setToolTip(caption)
        self.setSelectionMode(QListWidget.NoSelection)
        self.setMaximumHeight(108)
        self._filling = False
        self.itemChanged.connect(self._item_changed)

    def _item_changed(self, _item) -> None:
        if not self._filling:
            self.changed.emit()

    def options(self) -> list:
        return [self.item(row).data(Qt.UserRole) for row in range(self.count())]

    def values(self) -> tuple:
        """What is ticked, in the offered order."""
        return tuple(self.item(row).data(Qt.UserRole)
                     for row in range(self.count())
                     if self.item(row).checkState() == Qt.Checked)

    def setOptions(self, options) -> None:  # noqa: N802 - Qt naming
        """Offer these, keeping whatever of them was already ticked."""
        self.setValues(self.values(), options)

    def setValues(self, values, options=None) -> None:  # noqa: N802
        """Tick exactly ``values``.

        A value that is not on the offered list is ADDED to it rather than
        dropped: a style file naming a compartment this screen has none of
        would otherwise lose it the moment the panel was refilled.
        """
        wanted = list(dict.fromkeys(values or ()))
        offered = list(self.options() if options is None else options)
        offered += [value for value in wanted if value not in offered]
        self._filling = True
        try:
            self.clear()
            for option in offered:
                item = QListWidgetItem(str(option), self)
                item.setData(Qt.UserRole, option)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if option in wanted
                                   else Qt.Unchecked)
        finally:
            self._filling = False


class _ReadOnlyValue(QLabel):
    """A setting the panel shows and does not edit.

    The two the volcano has -- the split axis's pair of pairs and the
    per-point annotation map -- are set from the plot itself. Showing them
    greyed is the same rule the right-click menu follows: a setting silently
    absent from the panel is one the user has been told exists and cannot
    find.
    """

    def show_value(self, value) -> None:
        text = "none" if value in (None, (), {}, "") else str(value)
        self.setText(text)


class VolcanoExplorer(QWidget):
    """An interactive volcano: click a point, restyle it, export it."""

    point_selected = Signal(dict)
    style_changed = Signal()

    def __init__(self, results: pd.DataFrame | None = None,
                 style: VolcanoStyle | None = None, parent=None):
        super().__init__(parent)
        self._results = pd.DataFrame() if results is None else results.reset_index(drop=True)
        self._style = style or VolcanoStyle()
        self._controls: dict[str, QWidget] = {}
        #: The name beside each control, so a broken setting can be pointed
        #: at. Held rather than looked up off the form every time: a red
        #: label has to be cleared again when the value is corrected, and
        #: that means knowing every label, not only the offending one.
        self._labels: dict[str, QLabel] = {}
        #: Which folding section holds each setting, so one that goes red
        #: can open itself. A red label inside a closed section is a red
        #: label nobody sees.
        self._sections: dict[str, QWidget] = {}
        #: The last value of each setting that actually drew, and the whole
        #: style it drew from. The first is what a broken setting falls back
        #: to; the second is the safety net for a failure no single setting
        #: can be blamed for.
        self._last_good: dict[str, Any] = {}
        self._last_good_style: VolcanoStyle | None = None
        self._problems: dict[str, str] = {}
        self._problem_ink = self._error_ink()
        self._updating = False
        self._selected_index: int | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal, self)
        outer.addWidget(splitter)

        # --- canvas -------------------------------------------------------
        from .graph_builder import _canvas_class
        from matplotlib.figure import Figure

        self._figure = Figure(figsize=(self._style.figure_width,
                                       self._style.figure_height))
        self._canvas = _canvas_class()(self._figure)
        self._canvas.setMinimumSize(420, 320)
        self._canvas.mpl_connect("button_press_event", self._on_click)
        # INSTRUCTION 108: RIGHT-CLICK THE FIGURE ITSELF. Every control this
        # explorer offers is in the side panel, which is the right home for
        # them -- but 108 is about reaching a figure's own style FROM the
        # figure, and a matplotlib canvas had no menu at all. The entries are
        # built from `dataclasses.fields(VolcanoStyle)` by the same two
        # functions the pyqtgraph plots use, so a style that gains a field
        # gains a menu entry here without anyone remembering to add one.
        self._canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        self._canvas.customContextMenuRequested.connect(self._style_menu)

        # THE EXPLANATION GOES UNDER THE PLOT. Not over it and not instead
        # of it: a message that replaces the figure takes away the one thing
        # the reader was looking at, over a single mistyped field.
        self._problem_line = QLabel("", self)
        self._problem_line.setObjectName("VolcanoProblems")
        self._problem_line.setWordWrap(True)
        self._problem_line.setVisible(False)

        left = QWidget(self)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self._canvas, 1)
        left_layout.addWidget(self._problem_line)
        left_layout.addWidget(self._build_detail_panel())
        splitter.addWidget(left)

        # --- controls -----------------------------------------------------
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(300)
        scroll.setWidget(self._build_controls())
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        self.setAcceptDrops(True)
        if not self._results.empty:
            self.refresh()
        # Hover help belongs on a setting's NAME, not on the field the user
        # is about to type into (instruction 113). One post-pass rather than
        # a convention every hand-built row has to remember.
        from ..screens.settings_model import retarget_field_tooltips
        retarget_field_tooltips(self)

    # ------------------------------------------------------------------ data

    def set_results(self, results: pd.DataFrame) -> None:
        self._results = results.reset_index(drop=True)
        self._selected_index = None
        self._repopulate_column_menus()
        self.refresh()

    def results(self) -> pd.DataFrame:
        return self._results.copy()

    def style(self) -> VolcanoStyle:
        return self._style

    def set_style(self, style: VolcanoStyle) -> None:
        self._style = style
        self._push_style_to_controls()
        self.refresh()

    def build_style_menu(self):
        """Build the context menu for styling the current volcano plot.

        Returns
        -------
        PySide6.QtWidgets.QMenu
            A menu containing live style controls and style-file actions.
            Changes update both the plot and its side-panel controls.
        """
        from PySide6.QtWidgets import QMenu

        from .fast_plots import add_style_entries, add_style_file_entries

        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        # ONE REDRAW PER CHANGE, and through `set_style` rather than
        # `refresh`, so the side panel's controls follow the menu. Two ways to
        # change one setting that disagree about what it now is would be worse
        # than having only one of them.
        def changed(_name=None, _value=None):
            self.set_style(self._style)

        add_style_entries(menu, self._style, changed,
                          choices=self._style_choices(),
                          labels=self._style_labels())
        menu.addSeparator()
        add_style_file_entries(menu, self._style, changed, parent=self)
        return menu

    def _style_choices(self) -> dict:
        """``{field: values}`` for every setting the side panel closes.

        READ OFF THE PANEL ITSELF rather than from a list of names. The list
        was written as ``colour_by`` and ``label_by`` while the controls --
        and :class:`VolcanoStyle` -- spell them ``color_by`` and
        ``label_column``, so those two never matched: the menu offered a text
        box where the panel offered a picker, and one of the two routes let a
        user type a column that does not exist. A panel control that IS a
        closed list is now the definition of one, so the two cannot disagree
        again and the marker, the colormap, the line style and the fonts get
        their pickers on the menu as well.
        """
        choices: dict = {}
        for name, widget in self._controls.items():
            if isinstance(widget, _MultiSelect):
                if widget.count():
                    choices[name] = list(widget.options())
                continue
            if not isinstance(widget, QComboBox):
                continue
            values = []
            for index in range(widget.count()):
                data = widget.itemData(index)
                # A "— none —" row carries `None` as its data, and `None` IS
                # the value there -- it is how a colour-by column is taken
                # back off -- so it stays on the offered list. Any other row
                # with no data falls back to what it says.
                if data is None and widget.itemText(index) != _NONE_ROW:
                    data = widget.itemText(index)
                values.append(data)
            if values:
                choices[name] = values
        return choices

    def _style_labels(self) -> dict:
        """``{field: {value: what the panel calls it}}``.

        The panel says "Circle" and the style stores ``"o"``; the menu has to
        say "Circle" too, or the two routes offer the same setting in two
        vocabularies and only one of them is the one a reader recognises.
        Only rows whose words differ from their value are listed, so the
        common case costs nothing.
        """
        named: dict = {}
        for name, widget in self._controls.items():
            if not isinstance(widget, QComboBox):
                continue
            rows = {}
            for index in range(widget.count()):
                data = widget.itemData(index)
                text = widget.itemText(index)
                if text != ("" if data is None else str(data)):
                    rows[data] = text
            if rows:
                named[name] = rows
        return named

    def menu_settings(self) -> set:
        """The style fields the right-click menu offers, by name."""
        from .fast_plots import style_menu_fields

        return style_menu_fields(self.build_style_menu())

    def panel_settings(self) -> set:
        """The style fields the side panel offers, by name."""
        return set(self._controls)

    def _style_menu(self, position) -> None:
        """Right-click on the canvas: build the menu and show it."""
        self.build_style_menu().exec(self._canvas.mapToGlobal(position))

    def merge_annotation_file(self, path, *, on: str | None = None) -> int:
        """Merge a CSV/Excel of annotations onto the results.

        Every column it brings becomes selectable for colour and shape. The
        join column is inferred from whichever shared column matches most
        rows, so a file keyed on ``gene`` and one keyed on ``guide`` both work
        without the user being asked which is which.

        :returns: the number of columns added.
        """
        path = os.fspath(path)
        frame = (pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls"))
                 else pd.read_csv(path))
        if frame.empty:
            return 0
        shared = [c for c in frame.columns if c in self._results.columns]
        if on is not None:
            key = on
        elif shared:
            # The column that actually matches the most rows, not the first
            # one that happens to share a name.
            key = max(shared, key=lambda c: self._results[c].astype(str).isin(
                frame[c].astype(str)).sum())
        else:
            raise ValueError(
                f"{os.path.basename(path)} shares no column with the results "
                f"({sorted(self._results.columns)[:10]}), so there is nothing "
                f"to join on.")
        incoming = frame.drop_duplicates(subset=[key])
        new_columns = [c for c in incoming.columns if c not in self._results.columns]
        if not new_columns:
            return 0
        # Join on the string form of the key, so 'TGGT1_225160' matches
        # whatever dtype each side happened to be read as -- a numeric-looking
        # guide id read as int on one side and str on the other would
        # otherwise match nothing and silently annotate every row with NaN.
        merged = self._results.copy()
        lookup = incoming.set_index(incoming[key].astype(str))
        keys = self._results[key].astype(str)
        for column in new_columns:
            merged[column] = keys.map(lookup[column])
        self._results = merged
        self._repopulate_column_menus()
        self.refresh()
        return len(new_columns)

    # -------------------------------------------------------------- controls

    def _build_controls(self) -> QWidget:
        """Build the volcano controls in four initially collapsed sections.

        The significance level, threshold method, threshold multiplier, and
        colour-by field remain visible because they define the primary
        statistical filtering and visual encoding choices. Plot, test,
        appearance, and label options are grouped in collapsible sections.
        """
        from .section import Section

        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        constant = QWidget(container)
        constant_form = QFormLayout(constant)
        constant_form.setContentsMargins(8, 8, 8, 8)
        constant_form.setSpacing(4)
        for key, widget in (
            ("alpha", self._spin(1e-6, 0.5, 0.01, 6, "Significance level")),
            ("threshold_method", self._combo(
                ["value", "std", "mad", "quantile", "control"],
                "How the effect-size cut is derived")),
            ("threshold_multiplier", self._spin(
                0.0, 100.0, 0.5, 4,
                "Multiplier applied to the rule above (the quantile itself "
                "when the method is 'quantile')")),
            ("color_by", self._combo([], "Column that chooses each colour")),
        ):
            constant_form.addRow(self._register(key, widget), widget)
        layout.addWidget(constant)

        for title, rows in (
            ("What is plotted", [
                ("x_column", self._combo([], "Column plotted on the x axis")),
                ("y_column", self._combo([], "Column plotted on the y axis")),
                ("y_neg_log10", self._check("\u2212log\u2081\u2080 the y column")),
                ("label_column", self._combo([], "Column holding point names")),
                ("x_scale", self._combo(SCALES, "X axis scale")),
                ("y_scale", self._combo(SCALES, "Y axis scale")),
                ("x_lim", self._optional(2, -1e9, 1e9, 4, "X axis limits")),
                ("y_lim", self._optional(2, -1e9, 1e9, 4, "Y axis limits")),
                ("invert_x", self._check("Invert x axis")),
                ("invert_y", self._check("Invert y axis")),
                ("split_axis", self._check("Split the y axis (broken axis)")),
                ("split_height_ratio", self._spin(
                    0.1, 0.9, 0.05, 2, "Height of the upper panel")),
                ("split_y_lims", self._readonly(
                    "Where the y axis is split (set by ticking the split "
                    "above)")),
            ]),
            ("How it is tested", [
                ("effect_threshold", self._optional(
                    1, -1e6, 1e6, 4,
                    "Effect-size cut used when the method is 'value'; "
                    "automatic draws no effect-size line at all")),
                ("control_column", self._combo(
                    [], "Boolean column marking the non-targeting controls, "
                        "for the 'control' method")),
                ("show_alpha_line", self._check("Draw the significance line")),
                ("show_effect_lines", self._check(
                    "Draw the effect-size lines")),
                ("show_zero_line", self._check("Draw the zero line")),
            ]),
            ("How it looks", [
                ("marker", self._combo(
                    [code for code, _ in MARKER_SHAPES], "Marker shape",
                    labels=[label for _, label in MARKER_SHAPES])),
                ("marker_size", self._spin(1, 400, 2, 1, "Marker size")),
                ("significant_marker_size", self._spin(
                    1, 400, 2, 1, "Marker size for significant points")),
                ("marker_alpha", self._spin(0.05, 1.0, 0.05, 2, "Opacity")),
                ("marker_edge_width", self._spin(0, 5, 0.05, 2, "Edge width")),
                ("marker_edge_color", self._line("Edge colour")),
                ("base_color", self._line("Colour of non-significant points")),
                ("significant_color", self._line(
                    "Colour of significant points")),
                ("colormap", self._combo(
                    [name for group in COLORMAPS.values() for name in group],
                    "Colormap")),
                ("color_vmin", self._optional(1, -1e9, 1e9, 4,
                                              "Low end of the colour scale")),
                ("color_vmax", self._optional(1, -1e9, 1e9, 4,
                                              "High end of the colour scale")),
                ("shape_by", self._combo([], "Column that chooses each shape")),
                # SEVERAL AT ONCE. Ticking two compartments asks one question
                # about both, which is why this is a list of tick boxes and
                # not a drop-down.
                ("localizations", self._multi(
                    "Colour by localization \u2014 tick any combination")),
                ("localization_column", self._combo(
                    [], "Column naming each row's gene, for the compartment "
                        "lookup")),
                ("show_colorbar", self._check("Show the colour bar")),
                ("line_width", self._spin(0, 10, 0.1, 2,
                                          "Threshold line width")),
                ("line_color", self._line("Threshold line colour")),
                ("line_style", self._combo(
                    [code for code, _ in LINE_STYLES], "Threshold line style",
                    labels=[label for _, label in LINE_STYLES])),
                ("zero_line_width", self._spin(0, 10, 0.1, 2,
                                               "Zero line width")),
                ("zero_line_color", self._line("Zero line colour")),
                ("grid", self._check("Show grid")),
                ("grid_axis", self._combo(["x", "y", "both", "none"],
                                          "Grid axis")),
                ("grid_color", self._line("Grid colour")),
                ("grid_width", self._spin(0, 5, 0.1, 2, "Grid line width")),
                ("hide_top_right_spines", self._check(
                    "Hide top/right spines")),
                ("axis_color", self._line(
                    "Colour of the axis lines, the ticks and the text")),
                ("screen_background", self._line(
                    "Background on screen ('none' shows the page through)")),
                ("background_color", self._line(
                    "Background of an exported figure ('none' leaves the "
                    "page showing through)")),
                ("figure_width", self._spin(1, 40, 0.2, 2,
                                            "Figure width (in)")),
                ("figure_height", self._spin(1, 40, 0.2, 2,
                                             "Figure height (in)")),
                ("dpi", self._int_spin(50, 1200, "Raster export dpi")),
                ("legend", self._check("Show legend")),
                ("legend_location", self._combo(
                    ["best", "upper right", "upper left", "lower left",
                     "lower right", "right", "center left", "center right",
                     "lower center", "upper center", "center"],
                    "Legend position")),
                ("transparent", self._check(
                    "Transparent background on export")),
            ]),
            ("What is labelled", [
                ("title", self._line("Plot title")),
                ("x_label", self._line("X axis title")),
                ("y_label", self._line("Y axis title (blank = automatic)")),
                ("font_family", self._combo(FONT_FAMILIES, "Font family")),
                ("font_size", self._spin(4, 48, 0.5, 1, "Base font size")),
                ("title_font_size", self._spin(4, 48, 0.5, 1, "Title size")),
                ("label_font_size", self._spin(4, 48, 0.5, 1,
                                               "Annotation size")),
                ("tick_font_size", self._spin(4, 48, 0.5, 1,
                                              "Tick label size")),
                ("font_weight", self._combo(
                    ["normal", "bold", "light", "medium", "semibold",
                     "heavy"], "Font weight")),
                ("annotate_significant", self._check("Label every hit")),
                ("annotations", self._readonly(
                    "Points labelled by name (set by clicking a point)")),
            ]),
        ):
            section = Section(title, container)
            for key, widget in rows:
                section.add_row(self._register(key, widget, section), widget)
            layout.addWidget(section)

        buttons = QWidget(self)
        row = QVBoxLayout(buttons)
        row.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        for text, slot, tip in (
            ("Export PDF\u2026", lambda: self.export("pdf"),
             "Write a vector PDF of exactly this plot"),
            ("Export PNG\u2026", lambda: self.export("png"),
             "Write a raster PNG at the dpi set under Frame"),
        ):
            button = QPushButton(text, self)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            top.addWidget(button)
        row.addLayout(top)
        bottom = QHBoxLayout()
        for text, slot, tip in (
            ("Load annotations\u2026", self._pick_annotation_file,
             "Merge a CSV/Excel of annotations, then colour or shape by any "
             "of its columns"),
            ("Save style", self._save_style, "Save this appearance as JSON"),
            ("Load style", self._load_style, "Restore a saved appearance"),
        ):
            button = QPushButton(text, self)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            bottom.addWidget(button)
        row.addLayout(bottom)
        layout.addWidget(buttons)
        layout.addStretch(1)
        return container

    def _register(self, key: str, widget: QWidget, section=None) -> QLabel:
        """Record a control, and build the name that sits beside it.

        THE LABEL IS BUILT HERE RATHER THAN BY THE FORM. ``addRow`` with a
        string makes a QLabel nobody holds a reference to, and every one of
        them is needed back -- to turn red when its setting breaks, and,
        just as much, to turn black again when it is corrected.
        """
        self._controls[key] = widget
        label = QLabel(str(widget.property("caption") or key), self)
        # A CAPTION IS A SENTENCE HERE, not a word: "Multiplier applied to
        # the rule above (the quantile itself when the method is
        # 'quantile')" is one of them. Unwrapped, a form layout gives the
        # name every pixel it asks for and pushes the field it names off the
        # side of the panel, which is a control the user cannot reach.
        label.setWordWrap(True)
        label.setMaximumWidth(190)
        self._labels[key] = label
        if section is not None:
            self._sections[key] = section
        return label

    # Each factory stores the caption on the widget so _group can label it
    # without a parallel table that can fall out of step.
    def _combo(self, options, caption: str, *, labels=None) -> QComboBox:
        widget = QComboBox(self)
        widget.setProperty("caption", caption)
        widget.setToolTip(caption)
        for index, option in enumerate(options):
            widget.addItem(labels[index] if labels else str(option), option)
        widget.currentIndexChanged.connect(self._on_control_changed)
        return widget

    def _check(self, caption: str) -> QCheckBox:
        widget = QCheckBox(self)
        widget.setProperty("caption", caption)
        widget.setToolTip(caption)
        widget.toggled.connect(self._on_control_changed)
        return widget

    def _spin(self, low, high, step, decimals, caption) -> QDoubleSpinBox:
        widget = QDoubleSpinBox(self)
        widget.setProperty("caption", caption)
        widget.setToolTip(caption)
        widget.setRange(low, high)
        widget.setSingleStep(step)
        widget.setDecimals(decimals)
        widget.valueChanged.connect(self._on_control_changed)
        return widget

    def _int_spin(self, low, high, caption) -> QSpinBox:
        widget = QSpinBox(self)
        widget.setProperty("caption", caption)
        widget.setToolTip(caption)
        widget.setRange(low, high)
        widget.valueChanged.connect(self._on_control_changed)
        return widget

    def _line(self, caption: str) -> QLineEdit:
        widget = QLineEdit(self)
        widget.setProperty("caption", caption)
        widget.setToolTip(caption)
        widget.editingFinished.connect(self._on_control_changed)
        return widget

    def _optional(self, count, low, high, decimals, caption):
        widget = _OptionalNumbers(count, low, high, decimals, caption, self)
        widget.setProperty("caption", caption)
        widget.setToolTip(caption)
        widget.changed.connect(self._on_control_changed)
        return widget

    def _multi(self, caption: str) -> "_MultiSelect":
        widget = _MultiSelect(caption, self)
        widget.setProperty("caption", caption)
        widget.changed.connect(self._on_control_changed)
        return widget

    def _readonly(self, caption: str) -> "_ReadOnlyValue":
        widget = _ReadOnlyValue("none", self)
        widget.setProperty("caption", caption)
        widget.setToolTip(caption)
        widget.setEnabled(False)
        return widget

    def _build_detail_panel(self) -> QWidget:
        box = QGroupBox("Selected point", self)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 8, 8, 8)
        self._detail_hint = QLabel("Click any point to see everything known "
                                   "about it.", self)
        self._detail_hint.setWordWrap(True)
        layout.addWidget(self._detail_hint)
        self._detail_table = QTableWidget(0, 2, self)
        install_sorting(self._detail_table)
        self._detail_table.setHorizontalHeaderLabels(["Field", "Value"])
        self._detail_table.verticalHeader().setVisible(False)
        self._detail_table.horizontalHeader().setStretchLastSection(True)
        self._detail_table.setMaximumHeight(190)
        layout.addWidget(self._detail_table)
        box.setMaximumHeight(260)
        return box

    # --------------------------------------------------------------- syncing

    def _repopulate_column_menus(self) -> None:
        """Refill the column dropdowns from whatever columns now exist."""
        columns = list(self._results.columns)
        numeric = [c for c in columns
                   if pd.api.types.is_numeric_dtype(self._results[c])]
        mappable = [c for c in columns if c not in _NON_MAPPING_COLUMNS]
        self._updating = True
        try:
            for key, options, allow_none in (
                ("x_column", numeric, False),
                ("y_column", numeric, False),
                ("label_column", columns, False),
                ("color_by", mappable, True),
                ("shape_by", mappable, True),
                ("control_column", columns, True),
                ("localization_column", columns, True),
            ):
                widget = self._controls.get(key)
                if widget is None:
                    continue
                current = widget.currentData()
                widget.clear()
                if allow_none:
                    widget.addItem(_NONE_ROW, None)
                for option in options:
                    # pandas 3 normalizes a ``None`` column label to ``nan``.
                    # Keep the menu contract stable: an unnamed column is
                    # shown as "None" and, unlike the em-dash sentinel above,
                    # falls back to that visible text in ``_style_choices``.
                    try:
                        missing = pd.isna(option)
                    except (TypeError, ValueError):
                        missing = False
                    if isinstance(missing, (bool, np.bool_)) and missing:
                        widget.addItem("None", None)
                    else:
                        widget.addItem(str(option), option)
                index = widget.findData(current)
                if index < 0:
                    index = widget.findData(getattr(self._style, key, None))
                widget.setCurrentIndex(max(index, 0))
            compartments = self._controls.get("localizations")
            if compartments is not None:
                compartments.setOptions(self.compartments())
        finally:
            self._updating = False

    def compartments(self) -> list:
        """The LOPIT compartments this screen actually has, commonest first.

        NOT ALL 27 IN THE REFERENCE TABLE: a tick box that would colour
        nothing is indistinguishable from a broken one. Empty when no column
        of the results names a gene, which is a volcano without compartment
        colouring rather than an error.
        """
        if self._results.empty:
            return []
        try:
            return localizations_present(self._results, self._style)
        except Exception:  # noqa: BLE001 - no reference table, no colouring
            return []

    def _push_style_to_controls(self) -> None:
        """Write the style into every control without triggering a redraw."""
        self._updating = True
        try:
            for key, widget in self._controls.items():
                value = getattr(self._style, key, None)
                if isinstance(widget, _MultiSelect):
                    widget.setValues(value or ())
                elif isinstance(widget, _OptionalNumbers):
                    widget.setValue(value)
                elif isinstance(widget, _ReadOnlyValue):
                    widget.show_value(value)
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                elif isinstance(widget, QComboBox):
                    index = widget.findData(value)
                    if index < 0:
                        index = widget.findText(str(value))
                    widget.setCurrentIndex(max(index, 0))
                elif isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                    if value is not None:
                        widget.setValue(value)
                elif isinstance(widget, QLineEdit):
                    widget.setText("" if value is None else str(value))
        finally:
            self._updating = False

    def _pull_style_from_controls(self) -> None:
        for key, widget in self._controls.items():
            if isinstance(widget, _ReadOnlyValue):
                # Shown, not edited: reading a label back would write its
                # printed form over the value it was printed from.
                continue
            if isinstance(widget, _MultiSelect):
                setattr(self._style, key, tuple(widget.values()))
            elif isinstance(widget, _OptionalNumbers):
                setattr(self._style, key, widget.value())
            elif isinstance(widget, QCheckBox):
                setattr(self._style, key, bool(widget.isChecked()))
            elif isinstance(widget, QComboBox):
                index = widget.currentIndex()
                setattr(self._style, key,
                        widget.itemData(index) if index >= 0 else None)
            elif isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                setattr(self._style, key, widget.value())
            elif isinstance(widget, QLineEdit):
                text = widget.text().strip()
                setattr(self._style, key, text)

    def _on_control_changed(self, *_args) -> None:
        if self._updating:
            return
        self._pull_style_from_controls()
        self.refresh()
        self.style_changed.emit()

    # -------------------------------------------------------------- painting

    def refresh(self) -> None:
        """Redraw the canvas, and name the settings that could not be used.

        A BAD SETTING DOES NOT COST THE READER THE PICTURE. One mistyped
        colour used to replace the whole volcano with the words "cannot draw
        this plot", which takes away the only thing on screen over a single
        field and does not say which field. Instead the offending settings
        fall back to the last value that drew, the figure stays up, the name
        of each offending setting goes red and the reasons are printed under
        the plot.

        ``screen=True``: this render is being read, so it takes
        :attr:`VolcanoStyle.screen_background`. The export path does not
        pass it and therefore keeps the transparent figure default.
        """
        if self._results.empty:
            return
        if not self._controls.get("x_column", QComboBox()).count():
            self._repopulate_column_menus()
            self._push_style_to_controls()
        self._derive_split_limits()
        problems = validate_style(self._results, self._style)
        for candidate in self._drawable_styles(problems):
            try:
                _figure, self._panels = render_volcano(
                    self._results, candidate, figure=self._figure,
                    screen=True)
            except Exception as error:  # noqa: BLE001 - never crash on style
                problems.setdefault(_setting_named_in(str(error)), str(error))
                continue
            self._remember(candidate)
            break
        else:
            # Nothing drew, not even the defaults: there has never been a
            # good style to fall back to. An empty frame rather than an
            # error over the plot -- the reasons are on the line under it,
            # and they are the same reasons either way.
            self._figure.clear()
            axis = self._figure.add_subplot(111)
            axis.set_axis_off()
            self._panels = [axis]
        self._show_problems(problems)
        self._canvas.draw_idle()

    def _derive_split_limits(self) -> None:
        """A split axis needs limits; derive them rather than refuse to draw.

        Wrapped, because the suggestion is read off the y column and the y
        column is one of the settings that can be broken. A split that
        cannot be sized is left unset, and the y column's own complaint is
        the one the reader is shown.
        """
        if not self._style.split_axis or self._style.split_y_lims:
            return
        try:
            self._style.split_y_lims = self._suggest_split()
        except Exception:  # noqa: BLE001 - the y column answers for this
            self._style.split_y_lims = None

    def _drawable_styles(self, problems: dict):
        """Styles to try, best first, so something stays on the canvas.

        1. the current style with each broken setting replaced by the last
           value of it that drew -- which is what "keep the figure, minus
           that setting's contribution" means in practice;
        2. failing that, the whole of the last style that drew;
        3. failing that, the plain defaults.

        Yielded rather than returned so the common case -- nothing is wrong
        -- builds exactly one style and copies nothing.
        """
        fallback = VolcanoStyle()
        if problems:
            remembered = {
                name: self._last_good.get(name, getattr(fallback, name))
                for name in problems if hasattr(fallback, name)}
            yield dataclasses.replace(self._style, **remembered)
        else:
            yield self._style
        if self._last_good_style is not None:
            yield self._last_good_style
        yield fallback

    def _remember(self, style: VolcanoStyle) -> None:
        """Record the style that just drew, field by field and whole."""
        self._last_good = {field.name: getattr(style, field.name)
                           for field in dataclasses.fields(style)}
        self._last_good_style = dataclasses.replace(style)

    # -------------------------------------------------------- broken settings

    @staticmethod
    def _error_ink() -> str:
        """The application's own error red, or the house one without a theme."""
        try:
            from ..theme import active_palette

            return str(active_palette().get("error") or _PROBLEM_INK)
        except Exception:  # noqa: BLE001 - a bare widget has no palette
            return _PROBLEM_INK

    def problems(self) -> dict:
        """Return validation problems recorded during the latest redraw."""
        return dict(self._problems)

    def label_for(self, setting: str) -> QLabel | None:
        """Return the label associated with a style setting."""
        return self._labels.get(setting)

    def section_for(self, setting: str):
        """Return the collapsible section containing a style setting."""
        return self._sections.get(setting)

    def sections(self) -> list:
        """Return unique settings sections in panel order."""
        seen: list = []
        for section in self._sections.values():
            if section not in seen:
                seen.append(section)
        return seen

    def _caption_of(self, setting: str) -> str:
        widget = self._controls.get(setting)
        caption = widget.property("caption") if widget is not None else None
        return str(caption or setting or "")

    def _show_problems(self, problems: dict) -> None:
        """Redden the offending names, open their sections, print the reasons.

        EVERY LABEL IS VISITED, not only the broken ones: clearing the red
        when a value is corrected is half the job, and a pass that touched
        only the offenders could never do it.
        """
        self._problems = dict(problems)
        ink = self._problem_ink
        for name, label in self._labels.items():
            broken = name in problems
            label.setProperty("volcanoProblem", broken)
            label.setStyleSheet(f"color: {ink};" if broken else "")
            if not broken:
                continue
            section = self._sections.get(name)
            # A RED LABEL INSIDE A CLOSED SECTION IS A RED LABEL NOBODY SEES.
            if section is not None and not section.is_expanded():
                section.set_expanded(True)
        lines = []
        for name, message in problems.items():
            caption = self._caption_of(name) if name else ""
            lines.append(f"{caption}: {message}" if caption else str(message))
        self._problem_line.setStyleSheet(f"color: {ink};")
        self._problem_line.setText("\n".join(lines))
        self._problem_line.setVisible(bool(lines))

    def _suggest_split(self):
        """Split just above the null cloud, at the 99th percentile of y."""
        y = self._plotted_y()
        if y.size == 0:
            return None
        cut = float(np.nanquantile(y, 0.99))
        top = float(np.nanmax(y))
        if not np.isfinite(cut) or not np.isfinite(top) or top <= cut:
            return None
        pad = (top - cut) * 0.1 or 0.1
        return ((0.0, cut), (cut + pad * 0.5, top + pad))

    def _plotted_y(self) -> np.ndarray:
        raw = pd.to_numeric(self._results[self._style.y_column],
                            errors="coerce").to_numpy(float)
        if self._style.y_neg_log10:
            return -np.log10(np.clip(raw, np.finfo(float).tiny, None))
        return raw

    # -------------------------------------------------------------- clicking

    def _on_click(self, event) -> None:
        if event.inaxes is None or self._results.empty:
            return
        index = self.nearest_point(event.xdata, event.ydata, event.inaxes)
        if index is None:
            return
        self.select_point(index)

    def nearest_point(self, x: float, y: float, axes=None) -> int | None:
        """Index of the point closest to ``(x, y)`` in axis-normalised space.

        Normalising by each axis's visible range is what makes a click land on
        the point that LOOKS closest. Raw Euclidean distance in data units
        picks the wrong point whenever the axes have different scales, which
        on a volcano they always do.
        """
        if self._results.empty:
            return None
        xs = pd.to_numeric(self._results[self._style.x_column],
                           errors="coerce").to_numpy(float)
        ys = self._plotted_y()
        x_span = y_span = 1.0
        if axes is not None:
            low, high = axes.get_xlim()
            x_span = abs(high - low) or 1.0
            low, high = axes.get_ylim()
            y_span = abs(high - low) or 1.0
        distance = np.hypot((xs - x) / x_span, (ys - y) / y_span)
        if not np.isfinite(distance).any():
            return None
        index = int(np.nanargmin(distance))
        # Ignore clicks on empty space: 5% of the diagonal is about the radius
        # of a marker at default size.
        return index if distance[index] <= 0.05 else None

    def select_point(self, index: int) -> dict:
        """Select a point by row index and show everything known about it."""
        detail = point_details(self._results, index, self._style)
        self._selected_index = int(index)
        rows = [(k, v) for k, v in detail.items() if not k.startswith("_")]
        rows += [(k.lstrip("_") + " (plotted)", v)
                 for k, v in detail.items() if k.startswith("_")]
        self._detail_table.setRowCount(len(rows))
        for row, (key, value) in enumerate(rows):
            if isinstance(value, float):
                text = f"{value:.6g}"
            else:
                text = str(value)
            self._detail_table.setItem(row, 0, table_item(str(key)))
            item = table_item(text)
            item.setToolTip(text)
            self._detail_table.setItem(row, 1, item)
        name = detail.get(self._style.label_column, index)
        self._detail_hint.setText(f"Selected {name}")
        self.point_selected.emit(detail)
        return detail

    def selected_index(self) -> int | None:
        return self._selected_index

    # ---------------------------------------------------------------- export

    def export(self, fmt: str = "pdf", path: str | None = None) -> str | None:
        """Re-render at print size and write the file. Returns the path
        written, or ``None``.

        Not a screenshot: the figure is drawn again from the same style, so a
        PDF stays vector and a PNG honours the dpi under Frame regardless of
        how large the widget happens to be on screen.
        """
        if self._results.empty:
            return None
        if path is None:
            filters = {"pdf": "PDF (*.pdf)", "png": "PNG (*.png)",
                       "svg": "SVG (*.svg)"}
            path, _selected = QFileDialog.getSaveFileName(
                self, f"Export volcano as {fmt.upper()}",
                f"volcano.{fmt}", filters.get(fmt, "All files (*)"))
            if not path:
                return None
        if not path.lower().endswith(f".{fmt}"):
            path = f"{path}.{fmt}"
        from matplotlib.figure import Figure

        from ...plot import FIGURE_FORMATS

        figure = Figure(figsize=(self._style.figure_width,
                                 self._style.figure_height),
                        dpi=self._style.dpi)
        try:
            if str(fmt).lower() in FIGURE_FORMATS:
                render_volcano(self._results, self._style, figure=figure,
                               save_path=path)
                return path
            # SVG IS OFFERED HERE AND THE ONE WRITER CANNOT KEEP IT. The
            # renderer's `save_path` goes through `spacr.plot.save_figure`,
            # which writes PNG and PDF and CORRECTS the file's extension to
            # whichever it wrote -- so an .svg asked for here was written as a
            # PDF beside it and this handed back the .svg path, naming a file
            # that had never been created. `save_figure_as` is the writer that
            # already answers for the vector formats, print rule included, and
            # it reports the path it actually wrote.
            from .figure_settings import save_figure_as

            render_volcano(self._results, self._style, figure=figure)
            written = save_figure_as(self, figure, path)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Export failed", str(error))
            return None
        if not written:
            QMessageBox.warning(
                self, "Export failed",
                f"Nothing could be written to {path}.")
            return None
        return written

    def _save_style(self) -> str | None:
        path, _selected = QFileDialog.getSaveFileName(
            self, "Save plot style", "volcano_style.json", "JSON (*.json)")
        if not path:
            return None
        return self._style.save(path)

    def _load_style(self) -> str | None:
        path, _selected = QFileDialog.getOpenFileName(
            self, "Load plot style", "", "JSON (*.json)")
        if not path:
            return None
        self.set_style(VolcanoStyle.load(path))
        return path

    def _pick_annotation_file(self) -> None:
        path, _selected = QFileDialog.getOpenFileName(
            self, "Load an annotation table", "",
            "Tables (*.csv *.tsv *.txt *.xlsx);;All files (*)")
        if not path:
            return
        try:
            added = self.merge_annotation_file(path)
        except Exception as error:  # noqa: BLE001
            QMessageBox.warning(self, "Could not merge annotations", str(error))
            return
        QMessageBox.information(
            self, "Annotations merged",
            f"Added {added} column{'s' if added != 1 else ''}. They are now "
            f"available under Colour & shape mapping.")

    # ------------------------------------------------------------ drag/drop

    @staticmethod
    def _dropped_paths(event) -> list[str]:
        mime = event.mimeData()
        if not mime.hasUrls():
            return []
        return [url.toLocalFile() for url in mime.urls() if url.isLocalFile()]

    def dragEnterEvent(self, event):  # noqa: N802 - Qt name
        if self._dropped_paths(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):  # noqa: N802 - Qt name
        paths = self._dropped_paths(event)
        if not paths:
            event.ignore()
            return
        for path in paths:
            if path.lower().endswith(".json"):
                self.set_style(VolcanoStyle.load(path))
            else:
                try:
                    self.merge_annotation_file(path)
                except Exception:  # noqa: BLE001 - drop is best-effort
                    continue
        event.acceptProposedAction()


__all__ = ["VolcanoExplorer"]
