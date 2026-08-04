"""Prediction Profiler — interrogate a fitted model one input at a time.

A coefficient table says which terms matter. It does not say what the model
would predict for a well like yours, and it certainly does not say what
happens if this one gRNA's fraction doubles while everything else stays put.
That question — one input moving, the rest pinned where you chose — is what
this screen answers.

::

    inputs (ranked)        ┌─────────────────────────────────┐
    ▸ grna[233460_1] +2.4  │            ______/              │
      grna[239740_3] -1.8  │      _____/                     │
      grna[000000_2] +0.1  │  ___/                           │
                           └─────────────────────────────────┘
                           held:  grna[239740_3] ──●──── 0.31
                                  grna[000000_2] ●────── 0.02

The left column is :func:`spacr.profiler.sensitivity` — every input ranked by
how far it actually moves the prediction, not by its coefficient, because a
large coefficient on an input that never varies moves nothing. That ranking
is what makes a three-thousand-term design usable: it tells you which input
to open the profiler on.

**Nothing is re-fitted.** The screen reads a coefficient table a regression
run already wrote and wraps it in :class:`spacr.profiler.FittedLinear`, which
is *reading* the fit. A profiler that re-fits is showing a second model under
the first one's name, and on a penalised backend with ``alpha='auto'`` it is
not even the same model. A caller that has a live fitted object can hand it
straight to :meth:`ProfilerScreen.set_model` and skip the file entirely.

**The link is named, not guessed.** A coefficient table does not record which
inverse link produced it, and applying the wrong one draws a plausible curve
on the wrong scale. So the link is a control the user sets, it defaults to
identity, and the axis label always says which one is in force.

**Where there is no design, the assumption is stated.** Without the original
design matrix the observed range of each input is unknown, so the screen
sweeps 0-1 — the range a per-gRNA fraction lives in — and says so in the
status strip rather than implying it measured something.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...profiler import (LINKS, Profile, from_coefficients, profile,
                         response_scale, sensitivity)
from ..job_runner import JobRunner
from ..theme import SPACING, active_palette, pane_surface, register_widget_qss

__all__ = ["APP_KEY", "CurveCanvas", "ProfilerScreen", "curve_points",
           "make_profiler_screen", "register"]

#: The app key this screen is registered under.
APP_KEY = "profiler"

#: Sidebar / tile name.
APP_NAME = "Prediction Profiler"

#: One-line summary; the tooltip and status tip.
APP_DESCRIPTION = (
    "Move one input of a fitted model and watch the prediction move")

#: The paragraph under this app's header, handed to the seam as ``intro``.
APP_INTRO = (
    "Interrogate a fitted regression: sweep one input across its range, hold "
    "every other input wherever you choose, and see what the model predicts. "
    "The inputs are ranked by how far each one actually moves the prediction, "
    "so a design with thousands of gRNA terms still tells you which one to "
    "look at first. Nothing is re-fitted — the coefficients a run already "
    "wrote are the model — and the axis always says which scale it is on, "
    "because a probability, a rate and a hinge margin are not the same curve.")

#: Why there is no ``spacr-run profiler``; reaches ``cli.INTERACTIVE_ONLY``.
APP_CLI_NOTE = (
    "The Prediction Profiler is an interactive sweep of one model input; "
    "headless, call spacr.profiler.profile(model, design, variable) for the "
    "same curve and spacr.profiler.sensitivity(model, design) for the same "
    "ranking.")

#: "Prediction Profiler" in the nine non-English UI languages, in
#: :data:`spacr.qt.i18n.LANGUAGES` order after English — sv, de, es, zh_CN,
#: pt, hi, ko, is, fr.
APP_TRANSLATIONS = (
    "Prediktionsprofilerare",
    "Vorhersage-Profiler",
    "Perfilador de predicciones",
    "预测剖析器",
    "Analisador de previsões",
    "पूर्वानुमान प्रोफ़ाइलर",
    "예측 프로파일러",
    "Spágreinir",
    "Profileur de prédiction",
)

#: The range each input is swept over when no design matrix is available.
#: A spaCR design column is a per-well gRNA fraction, which lives in [0, 1].
DEFAULT_RANGE: Tuple[float, float] = (0.0, 1.0)

#: How many held-value sliders to draw. The ranking decides which ones; a
#: design can have thousands of terms and a scroll area with thousands of
#: sliders in it is not a control, it is a wall.
MAX_SLIDERS = 8

#: How many points the swept curve has.
CURVE_POINTS = 61

#: Slider resolution. Integer steps, mapped onto the input's range.
SLIDER_STEPS = 200


def _profiler_qss(palette: dict, opacity) -> str:
    """QSS for the plot frame and the held-value panel."""
    surface = pane_surface("surface_alt", palette["theme"], opacity)
    return f"""
QFrame#ProfilerPlot {{
    background: {surface};
    border: 1px solid {palette["border_soft"]};
    border-radius: 8px;
}}
QScrollArea#ProfilerHeld {{
    background: {surface};
    border: 1px solid {palette["border_soft"]};
    border-radius: 8px;
}}
QLabel#ProfilerStatus[problem="true"] {{
    color: {palette["warning"]};
}}
"""


# ``replace=True`` because this module owns the name: a reimport must
# re-register the same block rather than raise and leave the screen unstyled.
register_widget_qss("ProfilerPlot", _profiler_qss, replace=True)


# ---------------------------------------------------------------------------
# The curve, as a pure function
# ---------------------------------------------------------------------------

def curve_points(curve: Optional[Profile], width: int, height: int, *,
                 margin: int = 36) -> List[Tuple[float, float]]:
    """Map a profile onto pixel coordinates inside ``width`` x ``height``.

    Split out from the widget so the plot is testable without reading pixels
    back: the shape of the curve is a property of this function, and the
    ``paintEvent`` only strokes what it returns.

    :param curve: the profile to plot; ``None`` or empty gives ``[]``.
    :param width: canvas width in pixels.
    :param height: canvas height in pixels.
    :param margin: gutter reserved for the axes.
    :returns: ``(x, y)`` pairs, left to right, y measured downwards.
    """
    if curve is None or len(curve) < 2:
        return []
    xs = [float(v) for v in curve.values]
    ys = [float(p) for p in curve.predictions if math.isfinite(p)]
    if len(ys) != len(xs) or not ys:
        return []
    x_low, x_high = min(xs), max(xs)
    y_low, y_high = min(ys), max(ys)
    if x_high == x_low:
        x_high = x_low + 1.0
    if y_high == y_low:
        # A flat curve is a real answer ("this input does nothing"), and it
        # must be drawn along the middle rather than divided by zero.
        y_low, y_high = y_low - 0.5, y_high + 0.5
    plot_width = max(1, width - 2 * margin)
    plot_height = max(1, height - 2 * margin)
    points: List[Tuple[float, float]] = []
    for x_value, y_value in zip(xs, curve.predictions):
        if not math.isfinite(y_value):
            continue
        x = margin + (x_value - x_low) / (x_high - x_low) * plot_width
        y = margin + (1.0 - (y_value - y_low) / (y_high - y_low)) * plot_height
        points.append((x, y))
    return points


class CurveCanvas(QWidget):
    """Draws one :class:`~spacr.profiler.Profile`.

    :param parent: Qt parent.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._curve: Optional[Profile] = None
        self._message = "Load a coefficient table to profile a model."
        self.setMinimumSize(360, 240)

    def set_curve(self, curve: Optional[Profile], message: str = "") -> None:
        """Show ``curve``; ``None`` shows ``message`` instead."""
        self._curve = curve
        if message:
            self._message = message
        self.update()

    def curve(self) -> Optional[Profile]:
        """The profile currently drawn, or ``None``."""
        return self._curve

    def points(self) -> List[Tuple[float, float]]:
        """The pixel coordinates the curve is currently drawn at."""
        return curve_points(self._curve, self.width(), self.height())

    def paintEvent(self, event) -> None:           # noqa: N802 - Qt override
        """Axes, then the curve, then the labels."""
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            palette = active_palette()
            if self._curve is None or len(self._curve) < 2:
                painter.setPen(QPen(QColor(palette["fg_muted"])))
                painter.drawText(
                    self.rect().adjusted(18, 18, -18, -18),
                    int(Qt.AlignTop | Qt.AlignLeft | Qt.TextWordWrap),
                    self._message)
                return

            margin = 36
            frame = QRect(margin, margin, max(1, self.width() - 2 * margin),
                          max(1, self.height() - 2 * margin))
            painter.setPen(QPen(QColor(palette["border_soft"])))
            painter.drawLine(frame.left(), frame.bottom(), frame.right(),
                             frame.bottom())
            painter.drawLine(frame.left(), frame.top(), frame.left(),
                             frame.bottom())

            path = QPainterPath()
            for index, (x, y) in enumerate(self.points()):
                if index == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            pen = QPen(QColor(palette["accent"]))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)

            painter.setPen(QPen(QColor(palette["fg_muted"])))
            painter.drawText(frame.left(), frame.bottom() + 22,
                             f"{self._curve.variable}  "
                             f"{self._curve.values[0]:.3g} → "
                             f"{self._curve.values[-1]:.3g}")
            painter.drawText(margin, margin - 12, self._curve.scale)
            painter.drawText(
                frame.right() - 200, margin - 12,
                f"{self._curve.predictions[0]:.4g} → "
                f"{self._curve.predictions[-1]:.4g}")
        finally:
            painter.end()


# ---------------------------------------------------------------------------
# The screen
# ---------------------------------------------------------------------------

class ProfilerScreen(QWidget):
    """Sweep one input of a fitted model; hold the rest.

    :param parent: Qt parent.
    :param coefficients: open straight onto this coefficient CSV.
    :param model: use this already-fitted object instead of reading a file.
    :param design: the design matrix, when the caller has it. Without one the
        screen sweeps :data:`DEFAULT_RANGE` and says so.
    :param threaded: ``False`` computes inline, so a test drives the screen
        synchronously without the behaviour diverging.
    :ivar last_error: text of the most recent failure, ``""`` when the last
        operation worked.
    """

    #: Emitted with the ranked :class:`~spacr.profiler.Sensitivity` list
    #: whenever a model is loaded.
    model_loaded = Signal(object)
    #: Emitted with the :class:`~spacr.profiler.Profile` after every redraw.
    profiled = Signal(object)

    def __init__(self, parent=None, coefficients: str = "",
                 model: Any = None, design: Optional[pd.DataFrame] = None,
                 threaded: bool = True):
        super().__init__(parent)
        self._model: Any = None
        self._design: Optional[pd.DataFrame] = design
        self._ranked: List[Any] = []
        self._curve: Optional[Profile] = None
        self._held: Dict[str, float] = {}
        self._sliders: Dict[str, QSlider] = {}
        self._slider_ranges: Dict[str, Tuple[float, float]] = {}
        self._slider_labels: Dict[str, QLabel] = {}
        self._jobs = JobRunner(self, threaded=threaded, app_key=APP_KEY)
        self._jobs.job_failed.connect(self._on_job_failed)
        self.last_error: str = ""

        self._build_ui()
        if model is not None:
            self.set_model(model, design=design)
        elif coefficients:
            self.load_coefficients(coefficients)
        else:
            self._set_status(
                "Choose a regression results.csv — its coefficients are the "
                "model.", problem=False)

    # -- construction -----------------------------------------------------

    def _build_ui(self) -> None:
        """Picker, status strip, then the ranked inputs beside the plot."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["lg"], SPACING["lg"],
                                 SPACING["lg"], SPACING["lg"])
        outer.setSpacing(SPACING["md"])

        title = QLabel(APP_NAME)
        title.setObjectName("ScreenTitle")
        outer.addWidget(title)

        subtitle = QLabel(
            "Move one input; hold the rest; see what the fitted model says.")
        subtitle.setObjectName("Muted")
        subtitle.setWordWrap(True)
        outer.addWidget(subtitle)

        picker = QHBoxLayout()
        picker.setSpacing(SPACING["sm"])
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText(
            "Coefficient table (results.csv from a regression run)")
        self._path_edit.returnPressed.connect(self._on_path_entered)
        picker.addWidget(QLabel("Model"))
        picker.addWidget(self._path_edit, 1)
        self._browse_button = QPushButton("Browse…")
        self._browse_button.clicked.connect(self._on_browse)
        picker.addWidget(self._browse_button)

        self._link = QComboBox()
        self._link.addItems(sorted(LINKS))
        self._link.setCurrentText("identity")
        self._link.setToolTip(
            "The inverse link the original fit used. A coefficient table does "
            "not record it, and applying the wrong one draws a plausible "
            "curve on the wrong scale — identity for OLS/WLS/RLM, logit or "
            "probit for the binomial fits, log for Poisson and horseshoe.")
        self._link.currentTextChanged.connect(self._on_link_changed)
        picker.addWidget(QLabel("Link"))
        picker.addWidget(self._link)

        self._points = QSpinBox()
        self._points.setRange(2, 501)
        self._points.setValue(CURVE_POINTS)
        self._points.setToolTip("How many points the swept curve has.")
        self._points.valueChanged.connect(self._on_control_changed)
        picker.addWidget(QLabel("Points"))
        picker.addWidget(self._points)
        outer.addLayout(picker)

        self._status = QLabel("")
        self._status.setObjectName("ProfilerStatus")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

        splitter = QSplitter(Qt.Horizontal)

        self._inputs = QTreeWidget()
        self._inputs.setHeaderLabels(["Input", "Coef.", "Moves by"])
        self._inputs.setRootIsDecorated(False)
        self._inputs.setMinimumWidth(260)
        self._inputs.currentItemChanged.connect(self._on_input_selected)
        splitter.addWidget(self._inputs)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(SPACING["sm"])

        plot_frame = QFrame()
        plot_frame.setObjectName("ProfilerPlot")
        plot_layout = QVBoxLayout(plot_frame)
        plot_layout.setContentsMargins(SPACING["sm"], SPACING["sm"],
                                       SPACING["sm"], SPACING["sm"])
        self._canvas = CurveCanvas()
        plot_layout.addWidget(self._canvas)
        right_layout.addWidget(plot_frame, 3)

        self._held_area = QScrollArea()
        self._held_area.setObjectName("ProfilerHeld")
        self._held_area.setWidgetResizable(True)
        self._held_area.setMinimumHeight(140)
        self._held_host = QWidget()
        self._held_layout = QVBoxLayout(self._held_host)
        self._held_layout.setContentsMargins(SPACING["sm"], SPACING["sm"],
                                             SPACING["sm"], SPACING["sm"])
        self._held_layout.setSpacing(SPACING["xs"])
        self._held_area.setWidget(self._held_host)
        right_layout.addWidget(self._held_area, 2)

        actions = QHBoxLayout()
        self._reset_button = QPushButton("Reset held values")
        self._reset_button.setToolTip(
            "Put every held input back at the median of the design (or at "
            "the middle of the assumed range when there is no design).")
        self._reset_button.clicked.connect(self._on_reset)
        actions.addWidget(self._reset_button)
        actions.addStretch(1)
        right_layout.addLayout(actions)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        outer.addWidget(splitter, 1)

    # -- loading ----------------------------------------------------------

    def load_coefficients(self, path: str) -> None:
        """Read a coefficient table and profile the model it describes."""
        path = str(path or "").strip()
        self.last_error = ""
        self._path_edit.setText(path)
        if not path:
            self._set_status("Choose a coefficient table.", problem=False)
            return
        link = self._link.currentText()
        self._set_status(f"Reading {os.path.basename(path) or path}…",
                         problem=False)
        self._jobs.cancel()
        self._jobs.submit(
            lambda target=path, chosen=link: from_coefficients(
                target, link=chosen,
                label=os.path.basename(os.path.dirname(target)) or "model"),
            self._on_model_ready)

    def set_model(self, model: Any, *,
                  design: Optional[pd.DataFrame] = None) -> None:
        """Profile an already-fitted object, skipping the file entirely."""
        if design is not None:
            self._design = design
        self._on_model_ready(model)

    def set_design(self, design: Optional[pd.DataFrame]) -> None:
        """Supply the design matrix, so the sweeps use observed ranges."""
        self._design = design
        if self._model is not None:
            self._on_model_ready(self._model)

    def model(self) -> Any:
        """The fitted object currently profiled, or ``None``."""
        return self._model

    def design(self) -> pd.DataFrame:
        """The design the sweeps run over — supplied or synthesized."""
        if self._design is not None and not self._design.empty:
            return self._design
        return self._synthetic_design()

    def _synthetic_design(self) -> pd.DataFrame:
        """A stand-in design over :data:`DEFAULT_RANGE` for each input.

        Two rows, at the low and high end, is all the profiler needs from a
        design: the sweep range and the median. Anything more would be
        inventing data.
        """
        if self._model is None:
            return pd.DataFrame()
        names = [str(name) for name in getattr(self._model, "params",
                                               pd.Series(dtype=float)).index]
        low, high = DEFAULT_RANGE
        columns: Dict[str, List[float]] = {}
        for name in names:
            if name.lower() in ("intercept", "const"):
                columns[name] = [1.0, 1.0]
            else:
                columns[name] = [low, high]
        return pd.DataFrame(columns)

    def _on_model_ready(self, model: Any) -> None:
        """Take a model, rank its inputs, draw the top one."""
        self._model = model
        if model is None:                             # pragma: no cover
            self._set_status("The model could not be read.", problem=True)
            return
        # A live fitted object carries its own link and applies it inside
        # predict(); the combo would be a control that changes nothing, which
        # is worse than no control. It is only meaningful for a model
        # rebuilt from a coefficient table, where the link is genuinely
        # unknown and has to be supplied.
        from ...profiler import FittedLinear

        rebuilt = isinstance(model, FittedLinear)
        self._link.setEnabled(rebuilt)
        self._link.setToolTip(
            self._link.toolTip() if rebuilt else
            f"This model carries its own link, applied on the "
            f"{response_scale(model)} scale. The setting only applies to a "
            f"model rebuilt from a written-out coefficient table.")
        design = self.design()
        if design.empty:
            self._set_status("That model has no inputs to profile.",
                             problem=True)
            self._inputs.clear()
            self._canvas.set_curve(None, "No inputs to profile.")
            return
        self._ranked = sensitivity(model, design)
        self._fill_inputs()
        self._build_sliders()
        self.model_loaded.emit(list(self._ranked))
        if self._ranked:
            self._inputs.setCurrentItem(self._inputs.topLevelItem(0))
            return
        # Two different nothings, and the difference is the fix: a model with
        # only an intercept was never going to be profilable, while a design
        # whose columns never vary is a design problem the user can solve.
        movable = [name for name in design.columns
                   if str(name).lower() not in ("intercept", "const")]
        reason = ("nothing to sweep: this model has only an intercept."
                  if not movable else
                  "nothing to sweep: every input is constant in this design, "
                  "so no value of it would change the prediction.")
        self._canvas.set_curve(None, reason.capitalize())
        self._set_status(f"There is {reason}", problem=True)

    def _fill_inputs(self) -> None:
        """Redraw the ranked input list."""
        self._inputs.clear()
        for record in self._ranked:
            coefficient = ("—" if math.isnan(record.coefficient)
                           else f"{record.coefficient:.4g}")
            item = QTreeWidgetItem([record.variable, coefficient,
                                    f"{record.span:.4g}"])
            item.setData(0, Qt.UserRole, record.variable)
            item.setToolTip(
                0, f"{record.variable}: sweeping {record.low:.4g} → "
                   f"{record.high:.4g} moves the prediction "
                   f"{record.prediction_low:.4g} → "
                   f"{record.prediction_high:.4g}.")
            self._inputs.addTopLevelItem(item)

    # -- held values ------------------------------------------------------

    def _build_sliders(self) -> None:
        """One slider per held input, for the most influential few."""
        while self._held_layout.count():
            item = self._held_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._sliders.clear()
        self._slider_labels.clear()
        self._slider_ranges.clear()
        self._held.clear()

        design = self.design()
        for record in self._ranked[:MAX_SLIDERS]:
            name = record.variable
            column = pd.to_numeric(design[name], errors="coerce").dropna()
            low = float(column.min()) if not column.empty else DEFAULT_RANGE[0]
            high = float(column.max()) if not column.empty else DEFAULT_RANGE[1]
            if not math.isfinite(low) or not math.isfinite(high) or low == high:
                low, high = DEFAULT_RANGE
            middle = float(column.median()) if not column.empty else (
                (low + high) / 2.0)
            self._slider_ranges[name] = (low, high)
            self._held[name] = middle

            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(SPACING["sm"])
            caption = QLabel(name)
            caption.setMinimumWidth(160)
            caption.setToolTip(name)
            layout.addWidget(caption)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, SLIDER_STEPS)
            slider.setValue(self._to_step(name, middle))
            slider.valueChanged.connect(
                lambda value, key=name: self._on_slider_moved(key, value))
            layout.addWidget(slider, 1)
            value_label = QLabel(f"{middle:.4g}")
            value_label.setMinimumWidth(70)
            layout.addWidget(value_label)
            self._held_layout.addWidget(row)
            self._sliders[name] = slider
            self._slider_labels[name] = value_label
        self._held_layout.addStretch(1)

    def _to_step(self, name: str, value: float) -> int:
        """Map a value onto the slider's integer scale."""
        low, high = self._slider_ranges.get(name, DEFAULT_RANGE)
        if high == low:
            return 0
        fraction = (float(value) - low) / (high - low)
        return int(round(min(1.0, max(0.0, fraction)) * SLIDER_STEPS))

    def _from_step(self, name: str, step: int) -> float:
        """Map a slider position back onto the input's range."""
        low, high = self._slider_ranges.get(name, DEFAULT_RANGE)
        return low + (high - low) * (int(step) / SLIDER_STEPS)

    def held_values(self) -> Dict[str, float]:
        """Where each held input currently sits."""
        return dict(self._held)

    def set_held(self, name: str, value: float) -> None:
        """Hold one input at ``value`` and redraw."""
        if name not in self._sliders:
            raise KeyError(f"{name!r} has no held-value control")
        self._sliders[name].setValue(self._to_step(name, value))

    def _on_slider_moved(self, name: str, step: int) -> None:
        """Record the new held value and redraw."""
        value = self._from_step(name, step)
        self._held[name] = value
        label = self._slider_labels.get(name)
        if label is not None:
            label.setText(f"{value:.4g}")
        self._redraw()

    def _on_reset(self) -> None:
        """Put every held input back at the middle of its range."""
        self._build_sliders()
        self._redraw()

    # -- profiling --------------------------------------------------------

    def variable(self) -> str:
        """The input currently swept, or ``""``."""
        item = self._inputs.currentItem()
        return str(item.data(0, Qt.UserRole)) if item is not None else ""

    def curve(self) -> Optional[Profile]:
        """The profile currently drawn, or ``None``."""
        return self._curve

    def ranked_inputs(self) -> List[Any]:
        """Every input, ranked by how far it moves the prediction."""
        return list(self._ranked)

    def _on_input_selected(self, *_args) -> None:
        """Sweep whichever input was selected."""
        self._redraw()

    def _on_control_changed(self, *_args) -> None:
        """Redraw after a control that changes the curve but not the model."""
        self._redraw()

    def _on_link_changed(self, *_args) -> None:
        """Re-read the coefficients under the newly chosen link."""
        if self._path_edit.text().strip():
            self.load_coefficients(self._path_edit.text())
        elif self._model is not None:
            self._redraw()

    def _redraw(self) -> None:
        """Recompute the curve and hand it to the canvas."""
        variable = self.variable()
        if self._model is None or not variable:
            return
        held = {name: value for name, value in self._held.items()
                if name != variable}
        try:
            self._curve = profile(
                self._model, self.design(), variable, at=held,
                n=int(self._points.value()))
        except (KeyError, ValueError, TypeError) as exc:
            self.last_error = str(exc)
            self._canvas.set_curve(None, str(exc))
            self._set_status(f"Could not profile {variable}: {exc}",
                             problem=True)
            return
        self._canvas.set_curve(self._curve)
        assumed = self._design is None or self._design.empty
        message = (
            f"{variable}: {self._curve.values[0]:.4g} → "
            f"{self._curve.values[-1]:.4g} moves the prediction "
            f"{self._curve.predictions[0]:.4g} → "
            f"{self._curve.predictions[-1]:.4g} "
            f"({response_scale(self._model)}), with "
            f"{len(held)} other input(s) held.")
        if assumed:
            message += (f" No design matrix was supplied, so every input is "
                        f"swept over the assumed range "
                        f"{DEFAULT_RANGE[0]:g}–{DEFAULT_RANGE[1]:g}.")
        self._set_status(message, problem=False)
        self.profiled.emit(self._curve)

    # -- slots ------------------------------------------------------------

    def _on_path_entered(self) -> None:
        """Load whatever was typed into the path box."""
        self.load_coefficients(self._path_edit.text())

    def _on_browse(self) -> None:                    # pragma: no cover - modal
        """Ask for a coefficient CSV and load it."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a coefficient table", "", "CSV (*.csv)")
        if path:
            self.load_coefficients(path)

    def _on_job_failed(self, message: str) -> None:
        """Report a background failure inline; never a modal."""
        self.last_error = message
        self._set_status(f"Could not read that model: {message}", problem=True)

    def _set_status(self, text: str, *, problem: bool) -> None:
        """Write the status strip and repolish it for the problem colour."""
        self._status.setText(text)
        self._status.setProperty("problem", "true" if problem else "false")
        style = self._status.style()
        if style is not None:
            style.unpolish(self._status)
            style.polish(self._status)

    # -- lifecycle --------------------------------------------------------

    def is_busy(self) -> bool:
        """True while a model is still being read."""
        return self._jobs.is_busy()

    def active_jobs(self) -> int:
        """How many worker threads are still winding down."""
        return self._jobs.active_jobs()

    def closeEvent(self, event) -> None:             # noqa: N802 - Qt override
        """Drain the worker before the widget goes."""
        self._jobs.shutdown()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def make_profiler_screen(app_key: Optional[str] = None) -> QWidget:
    """Factory the registry calls to build this screen."""
    return ProfilerScreen()


def register() -> bool:
    """Add Prediction Profiler to the app registry. Idempotent."""
    from ..app import APPS, SECTION_EXPLORE, STAGE_ALPHA, register_app

    if any(row[0] == APP_KEY for row in APPS):
        return False
    register_app(
        APP_KEY, APP_NAME, APP_DESCRIPTION, SECTION_EXPLORE,
        factory=make_profiler_screen, stage=STAGE_ALPHA,
        title="Prediction Profiler", intro=APP_INTRO, cli_note=APP_CLI_NOTE,
        api_module="qt/screens/profiler",
        translations=APP_TRANSLATIONS)
    return True


register()
