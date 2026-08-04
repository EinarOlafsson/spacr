"""
Power / Design — how many cells per well, and how many wells, do I need?

The science has been in the tree since the spaCRPower port landed:
:mod:`spacr.power_simulate` builds a screen you know the truth for, and
:mod:`spacr.power_model` fits the horseshoe-Poisson model to it and scores how
well the fit recovered the hits you planted. What has never existed is the
surface where a screener asks the question they actually have, which is not
"what is the AUROC of a design with well_abundance_factor_mu = 4.6" but::

    I have 452 genes and four 384-well plates. My classifier is about
    0.80 / 0.12. How many cells per well do I have to image before this
    screen finds its hits, and would more wells be cheaper than more cells?

Layout::

    ┌────────────────────────┬───────────────────────────────────────────────┐
    │ Library                │  At 123 cells per well and 4.6 constructs per │
    │  genes           [452] │  well, 1536 wells detect a 6.7-fold effect in │
    │  gRNAs/gene      [4]   │  67% of simulations — 2 of 3 replicates …     │
    │  score per    [gene ▾] │                                               │
    │  constructs/well [4.6] │  ! spaCRPower splits a well's cells evenly …  │
    │ Plate                  │  ! a replicate whose fit failed counts as …   │
    │  wells/plate  [384 ▾]  │  ! mean-field ADVI, not NUTS: the ranking …   │
    │  plates          [4]   ├───────────────────────────────────────────────┤
    │   → 1536 wells         │  detection probability vs cells per well      │
    │ Effect                 │   1 ┤        ╭──●───────                      │
    │  background     [0.12] │     │   ╭────╯                                │
    │  effect (fold)  [6.67] │   0 ┼───╯                                     │
    │   → 0.800 positive     │      15   31   62  123  246                   │
    │  prevalence    [0.025] ├───────────────────────────────────────────────┤
    │ …                      │  cells  power  mean AUROC  mean AP  n         │
    └────────────────────────┴───────────────────────────────────────────────┘

Design notes:

* **The screen computes no statistics of its own.** Every number on it comes
  out of :func:`spacr.power_model.scan_parameters`; the only arithmetic here
  is counting how many replicates cleared the user's threshold. The
  translation from "genes, plates, fold-change" to simulator keyword
  arguments lives in one function,
  :func:`spacr.qt.widgets.power_design.simulator_kwargs`, so a run started
  here and the same run typed at a Python prompt are the same call. The test
  suite asserts that byte for byte on a fixed seed.
* **The caveats are next to the number, not in a docstring.** The port
  departs from spaCRPower in ways that change the answer — most importantly
  that R's even cell-split *overstates* power — and a power analysis whose
  caveats are one import away is a power analysis that gets quoted without
  them. The ones that move the number are rendered beside the sentence; the
  rest are one click below it.
* **Off the GUI thread, and the thread actually retires.** A full sweep is
  minutes. It goes through :func:`spacr.qt.bridge.make_thread`, and every
  ``thread.finished`` slot is a BOUND METHOD — see
  :meth:`PowerScreen._retire_finished_jobs` for what a closure does here and
  why it is not a style preference.
* **No modal dialogs.** Every failure lands in the inline status line.
* **``threaded=False`` runs the same code inline**, firing the same signals,
  so the tested path and the shipped path differ only in where they run.
"""
from __future__ import annotations

import logging
import math
import re
import threading
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..theme import (SPACING, active_palette, make_transparent,
                     mark_surface, paint_panel, register_widget_qss)
from ..widgets.power_design import (
    CAVEATS,
    PLATE_FORMATS,
    DesignSpec,
    cells_grid,
    changes_the_number,
    estimate_runtime_s,
    plain_sentence,
    power_curve,
    simulator_kwargs,
    wells_grid,
)
from .app_screen import ModuleHeader

__all__ = [
    "APP_KEY",
    "CaveatPanel",
    "PowerCurveView",
    "PowerScreen",
    "make_power_screen",
    "power_default_settings",
    "register",
    "register_settings",
    "run_power_sweep",
]

LOG = logging.getLogger(__name__)

#: Stable app id. Chosen once; `bridge`, `cli` and saved user state key off it.
APP_KEY = "power"

#: objectName of the headline sentence, so the stylesheet can find it.
ANSWER_OBJECT = "spacrPowerAnswer"
#: objectName of the caveat panel.
CAVEAT_OBJECT = "spacrPowerCaveats"
#: objectName of the inline status line.
STATUS_OBJECT = "spacrPowerStatus"

#: Backends offered in the combo. "auto" first because it is the library's own
#: default and it says which one it picked; the rest are pinned choices for
#: when two machines have to agree.
_BACKEND_CHOICES = ("auto", "torch", "numpyro", "pymc")

#: Matches the progress line the job prints. Deliberately the same shape
#: ``bridge._PROGRESS_RE`` looks for, so one printed line drives both this
#: screen's bar and the Home screen's.
_PROGRESS_LINE = re.compile(r"\bProgress:\s*(\d+)\s*/\s*(\d+)")

#: Header of the per-point results table.
_TABLE_HEADERS = ("cells / well", "wells", "power", "detected",
                  "mean AUROC", "mean AP", "AP baseline", "not converged",
                  "failed")


#: The :class:`DesignSpec` fields the form owns. Everything else is *held*,
#: at the real screen's fitted value, and carried through
#: :attr:`PowerScreen._held` so a spec set programmatically survives a
#: round trip through the widget. Without that, ``set_spec`` would silently
#: reset the library skew and the classifier variances to their defaults and
#: a test — or a reloaded run — would be simulating a different screen from
#: the one it asked for.
_FORM_FIELDS = frozenset({
    "n_genes", "n_grnas_per_gene", "score_per", "cells_per_well",
    "wells_per_plate", "n_plates", "constructs_per_well",
    "background_positive_rate", "effect_fold", "hit_rate", "reads_per_well",
    "n_replicates", "detection_auroc", "seed", "backend",
})

#: The held fields, in the order the "held fixed" line lists them.
#:
#: Every :class:`DesignSpec` field the form does not ask for belongs here, or
#: ``set_spec`` stops round-tripping: ``spec()`` rebuilds the dataclass from
#: the form plus this dict, so a field in neither silently reverts to its
#: default and the sweep runs a different screen from the one that was loaded.
#: ``sequencing_error_rate`` and ``min_cells_per_well`` are listed for exactly
#: that reason -- both default to the spaCRPower behaviour, and a run that
#: turned either on has to still say so when it is reloaded.
_HELD_FIELDS = (
    "gene_abundance_alpha", "cells_per_well_var", "class_pos_var",
    "class_neg_var", "well_abundance_var", "sequencing_cells_per_well",
    "pcr_factor_mu", "pcr_factor_var", "read_depth_cv",
    "sequencing_error_rate", "min_cells_per_well", "imaging_split",
)


def _power_qss(palette: dict, opacity: Optional[float] = None) -> str:
    """QSS for this screen, contributed through :func:`register_widget_qss`.

    Scoped to this screen's object names so it cannot reach another app's
    labels. The caveat severities are a dynamic property rather than a
    per-label ``setStyleSheet`` for one reason that matters: an inline
    stylesheet is baked at construction and survives a theme switch, so a
    warning-orange caveat set under the dark theme stays dark-theme orange
    on a light background. Going through the registry means the colours are
    re-rendered from the live palette every time the stylesheet is rebuilt.

    :param palette: the live theme palette, surfaces already scrimmed.
    :param opacity: the user's page-opacity preference; unused — none of
        these rules paint a surface that should fade.
    :returns: the QSS block.
    """
    return f"""
#{ANSWER_OBJECT} {{
    color: {palette['fg']};
    background: {palette['surface_alt']};
    border: 1px solid {palette['border']};
    border-radius: 6px;
    padding: {SPACING['sm']}px;
}}
#{CAVEAT_OBJECT} QLabel[spacrCaveatSeverity="high"] {{
    color: {palette['warning']};
}}
#{CAVEAT_OBJECT} QLabel[spacrCaveatSeverity="note"] {{
    color: {palette['fg_muted']};
}}
#{STATUS_OBJECT} {{ color: {palette['fg_muted']}; }}
#{STATUS_OBJECT}[spacrError="true"] {{ color: {palette['error']}; }}
"""


# `replace=True` and at import: this module is reachable from two paths
# (the screens package and a direct import), and a second import must
# refresh the block rather than raise. Same posture as `pivot_builder`.
register_widget_qss("PowerDesign", _power_qss, replace=True)


# ---------------------------------------------------------------------------
# The job — a plain function, so it is testable without a QThread
# ---------------------------------------------------------------------------

def run_power_sweep(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run both sweeps for ``payload['spec']`` and put the result in ``payload``.

    Shaped as ``fn(settings)`` because that is what
    :func:`spacr.qt.bridge.make_thread` calls. Everything it needs is in the
    dict and everything it produces goes back into the same dict, so the GUI
    thread reads the result from an object it already owns rather than from a
    signal payload that would have to cross the thread boundary.

    Two separate calls to :func:`spacr.power_model.scan_parameters`, one per
    axis, rather than one call sweeping both: the Cartesian product of the two
    grids is 20 fits where the two questions need 9, and nobody asked what
    happens at a quarter of the wells *and* twice the cells.

    :param payload: mutable job dict with keys

        ``spec``
            the :class:`~spacr.qt.widgets.power_design.DesignSpec`.
        ``cancel``
            optional :class:`threading.Event`; set it and the sweep stops
            after the fit in flight, keeping the rows it already has.
        ``progress``
            optional ``fn(done, total, label)`` called on the worker thread.
        ``fit_kwargs``
            optional extra keyword arguments for the fit, for tests that
            need a sweep to finish in seconds.

    :returns: the result dict, which is also stored at ``payload['result']``:
        ``cells_scan``, ``wells_scan`` (raw
        :func:`~spacr.power_model.scan_parameters` frames), ``cells_curve``,
        ``wells_curve`` (from :func:`~spacr.qt.widgets.power_design.power_curve`),
        ``cancelled``, ``n_clipped_screens`` and ``clip_message``.
    """
    from ... import power_model as pm

    spec: DesignSpec = payload["spec"]
    cancel: Optional[threading.Event] = payload.get("cancel")
    progress: Optional[Callable[[int, int, str], None]] = payload.get("progress")
    fit_kwargs = dict(payload.get("fit_kwargs") or {})

    base = simulator_kwargs(spec)
    cells = cells_grid(spec)
    wells = wells_grid(spec)
    replicates = max(1, int(spec.n_replicates))
    total = (len(cells) + len(wells)) * replicates
    done = [0]

    def _make_hook(label: str):
        """A ``scan_parameters`` on_point that reports progress and cancels.

        Returns ``False`` — and only ``False`` — when the cancel event is
        set, which is the contract ``scan_parameters`` documents for
        stopping a sweep between points.
        """
        def _hook(point: Dict[str, Any]) -> Any:
            done[0] += 1
            if progress is not None:
                progress(done[0], total, label)
            if cancel is not None and cancel.is_set():
                return False
            return None
        return _hook

    # The spec travels WITH the result. Rendering against whatever is on the
    # form when the sweep lands would label a three-minute run with a design
    # the user edited while waiting for it.
    results: Dict[str, Any] = {"cancelled": False, "spec": spec}
    # `always`, not the default `once`: the abundance-clipping warning is
    # emitted from one source line, so the default filter would report the
    # first simulated screen that clipped and silently swallow the other
    # twenty-six — turning "most of this sweep clipped" into "one did".
    #
    # `catch_warnings` swaps the process-wide filter list, which is not
    # thread-safe. It is used anyway because only one sweep runs at a time
    # and the window is the sweep itself; the cost of getting it wrong is a
    # mis-counted clip warning, not a wrong power.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        cells_scan = pm.scan_parameters(
            **{**base, "imaging_n_cells_per_well_mu": cells},
            n_replicates=replicates,
            backend=str(spec.backend),
            seed=int(spec.seed),
            fit_kwargs=fit_kwargs or None,
            on_point=_make_hook("cells per well"),
        )
        results["cells_scan"] = cells_scan
        results["cancelled"] = bool(cells_scan.attrs.get("cancelled", False))

        if results["cancelled"]:
            wells_scan = cells_scan.iloc[0:0].copy()
        else:
            wells_scan = pm.scan_parameters(
                **{**base, "n_wells_per_screen": [int(w) for w in wells]},
                n_replicates=replicates,
                backend=str(spec.backend),
                seed=int(spec.seed),
                fit_kwargs=fit_kwargs or None,
                on_point=_make_hook("wells"),
            )
            results["cancelled"] = bool(wells_scan.attrs.get("cancelled", False))
        results["wells_scan"] = wells_scan

    clipped = [w for w in caught
               if w.category.__name__ == "AbundanceClippedWarning"]
    results["n_clipped_screens"] = len(clipped)
    results["clip_message"] = str(clipped[0].message) if clipped else ""

    results["cells_curve"] = power_curve(
        cells_scan, "imaging_n_cells_per_well_mu", spec.detection_auroc)
    results["wells_curve"] = power_curve(
        wells_scan, "n_wells_per_screen", spec.detection_auroc)
    payload["result"] = results
    return results


# ---------------------------------------------------------------------------
# The curve
# ---------------------------------------------------------------------------

class PowerCurveView(QWidget):
    """Detection probability against one swept axis, painted directly.

    QPainter rather than a matplotlib canvas: the plot is two axes, five
    points and a threshold line, and a FigureCanvas here would import
    matplotlib's Qt backend, own a timer and need the deleted-C++-object
    care :mod:`spacr.qt.widgets.umap_explorer` documents — for a drawing that
    is thirty lines of ``drawLine``.

    :meth:`describe` returns exactly what is drawn, as text, so a test can
    assert the content of the plot without reading pixels.
    """

    def __init__(self, title: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._title = str(title)
        self._points: List[Tuple[float, float]] = []
        self._counts: List[Tuple[int, int]] = []
        self._marker: Optional[float] = None
        self._threshold: float = 0.8
        self._x_label = ""
        self._palette = active_palette()
        # The panel drawn in `paintEvent` is the surface; the widget itself
        # must not also paint the blanket window fill underneath it.
        make_transparent(self)
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # -- content -----------------------------------------------------------

    def set_curve(self, curve, x_label: str, marker: Optional[float] = None,
                  threshold: float = 0.8) -> None:
        """Show ``curve``, a frame from :func:`power_design.power_curve`.

        :param curve: the curve, or ``None`` to clear.
        :param x_label: what the x axis counts.
        :param marker: x value of the user's own design, drawn as a rule so
            the point they came for is findable on their own plot.
        :param threshold: the detection AUROC, for the caption only.
        """
        self._points = []
        self._counts = []
        if curve is not None and len(curve):
            for _, row in curve.iterrows():
                self._points.append((float(row["value"]), float(row["power"])))
                self._counts.append((int(row["n_detected"]),
                                     int(row["n_replicates"])))
        self._x_label = str(x_label)
        self._marker = None if marker is None else float(marker)
        self._threshold = float(threshold)
        self.update()

    def describe(self) -> str:
        """The plotted values as one line of text, for tests and for export."""
        if not self._points:
            return f"{self._title}: no data"
        parts = [f"{x:g}={p:.2f} ({d}/{n})"
                 for (x, p), (d, n) in zip(self._points, self._counts)]
        return f"{self._title} [{self._x_label}]: " + ", ".join(parts)

    def is_empty(self) -> bool:
        """Whether there is anything to draw."""
        return not self._points

    # -- painting ----------------------------------------------------------

    def paintEvent(self, event):  # noqa: N802 - Qt name
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        palette = self._palette
        metrics = QFontMetrics(self.font())

        left = max(46, metrics.horizontalAdvance("100%") + 12)
        right = 10
        top = metrics.height() + 8
        bottom = metrics.height() * 2 + 10
        width = max(1, self.width() - left - right)
        height = max(1, self.height() - top - bottom)

        # A rounded translucent panel, not `fillRect(rect, surface)`: that
        # hex carries no alpha, so the fill was opaque by construction
        # whatever the page-opacity preference said, and the two curve views
        # were the only flat rectangles on a page of panels.
        paint_panel(painter, self, role="surface", inset=0.5)
        painter.setPen(QPen(QColor(palette["fg"]), 1))
        painter.drawText(6, metrics.ascent() + 2, self._title)

        if not self._points:
            painter.setPen(QPen(QColor(palette["fg_muted"]), 1))
            painter.drawText(self.rect(), Qt.AlignCenter,
                             "Press Run to draw this curve")
            painter.end()
            return

        xs = [x for x, _ in self._points]
        x_lo, x_hi = min(xs), max(xs)
        span = (x_hi - x_lo) or 1.0

        def to_px(x: float, y: float) -> Tuple[float, float]:
            return (left + width * (x - x_lo) / span,
                    top + height * (1.0 - max(0.0, min(1.0, y))))

        # y grid at 0 / 0.5 / 1, because a power curve is read against those
        # three and nothing else.
        painter.setPen(QPen(QColor(palette["border_soft"]), 1, Qt.DotLine))
        for level in (0.0, 0.5, 1.0):
            _, py = to_px(x_lo, level)
            painter.drawLine(int(left), int(py), int(left + width), int(py))
        painter.setPen(QPen(QColor(palette["fg_muted"]), 1))
        for level in (0.0, 0.5, 1.0):
            _, py = to_px(x_lo, level)
            painter.drawText(4, int(py + metrics.ascent() / 2 - 1),
                             f"{level * 100:.0f}%")

        if self._marker is not None and x_lo <= self._marker <= x_hi:
            painter.setPen(QPen(QColor(palette["accent"]), 1, Qt.DashLine))
            mx, _ = to_px(self._marker, 0.0)
            painter.drawLine(int(mx), int(top), int(mx), int(top + height))

        path = QPainterPath()
        for index, (x, y) in enumerate(self._points):
            px, py = to_px(x, y)
            if index == 0:
                path.moveTo(px, py)
            else:
                path.lineTo(px, py)
        painter.setPen(QPen(QColor(palette["accent"]), 2))
        painter.drawPath(path)

        painter.setBrush(QColor(palette["accent"]))
        for x, y in self._points:
            px, py = to_px(x, y)
            painter.drawEllipse(int(px) - 3, int(py) - 3, 6, 6)

        painter.setPen(QPen(QColor(palette["fg_muted"]), 1))
        baseline = self.height() - metrics.descent() - 2
        for x, _ in self._points:
            px, _py = to_px(x, 0.0)
            text = f"{x:g}"
            painter.drawText(int(px - metrics.horizontalAdvance(text) / 2),
                             int(baseline - metrics.height()), text)
        painter.drawText(int(left), int(baseline),
                         f"{self._x_label} — detection = AUROC ≥ "
                         f"{self._threshold:.2f}")
        painter.end()


# ---------------------------------------------------------------------------
# The caveat panel
# ---------------------------------------------------------------------------

class CaveatPanel(QWidget):
    """The port's departures from spaCRPower, rendered where they are read.

    The ones flagged ``changes_the_number`` are always visible; the rest are
    behind a "show all" toggle. That split is the panel's whole job: a list
    where the COM-Poisson third moment sits at the same weight as "the R
    version overstates power" is a list that gets skimmed, and the one line
    that would have changed somebody's plate count goes with it.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName(CAVEAT_OBJECT)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING["xs"])

        self._labels: List[QLabel] = []
        for caveat in changes_the_number():
            label = self._make_label(caveat, "high")
            layout.addWidget(label)
            self._labels.append(label)

        self._more = QPushButton("Show the rest of the caveats")
        self._more.setCheckable(True)
        self._more.setToolTip(
            "Departures from spaCRPower that do not change the power on "
            "screen, but do change how these numbers compare with the R "
            "package's.")
        self._more.toggled.connect(self._on_toggled)
        layout.addWidget(self._more, 0, Qt.AlignLeft)

        self._rest = QWidget()
        rest_layout = QVBoxLayout(self._rest)
        rest_layout.setContentsMargins(0, 0, 0, 0)
        rest_layout.setSpacing(SPACING["xs"])
        for caveat in CAVEATS:
            if caveat.changes_the_number:
                continue
            label = self._make_label(caveat, "note")
            rest_layout.addWidget(label)
            self._labels.append(label)
        self._rest.setVisible(False)
        layout.addWidget(self._rest)

    @staticmethod
    def _make_label(caveat, severity: str) -> QLabel:
        label = QLabel(f"! {caveat.headline}")
        label.setWordWrap(True)
        label.setToolTip(caveat.detail)
        label.setProperty("spacrCaveatSeverity", severity)
        label.setProperty("caveatKey", caveat.key)
        return label

    def _on_toggled(self, checked: bool) -> None:
        self._rest.setVisible(bool(checked))
        self._more.setText("Hide the rest of the caveats" if checked
                           else "Show the rest of the caveats")

    def visible_text(self) -> str:
        """Every caveat line currently on screen, joined. For tests."""
        return "\n".join(label.text() for label in self._labels
                         if label.isVisibleTo(self))

    def all_text(self) -> str:
        """Every caveat line the panel holds, shown or not."""
        return "\n".join(label.text() for label in self._labels)


# ---------------------------------------------------------------------------
# The screen
# ---------------------------------------------------------------------------

class PowerScreen(QWidget):
    """The Power / Design app.

    :param threaded: run sweeps on a :func:`~spacr.qt.bridge.make_thread`
        worker (the shipped behaviour). ``False`` runs the identical code
        inline and emits the identical signals, which is what the tests use
        to get a deterministic result without pumping an event loop.
    :param parent: Qt parent.
    """

    #: Emitted when a sweep settles, with whether it produced a result.
    job_finished = Signal(bool)
    #: Emitted on every progress tick, with ``(done, total)``.
    progressed = Signal(int, int)

    def __init__(self, threaded: bool = True,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._threaded = bool(threaded)
        self._jobs: List[Tuple[Any, Any]] = []
        self._pending: List[Dict[str, Any]] = []
        self._busy = False
        self._cancel: Optional[threading.Event] = None
        self._result: Optional[Dict[str, Any]] = None
        #: Extra fit arguments; tests lower the ADVI step count with this.
        self.fit_kwargs: Dict[str, Any] = {}
        #: The design fields that are not on the form, at the real screen's
        #: fitted values. Shown, never hidden — see ``_held_note``.
        self._held: Dict[str, Any] = {
            name: getattr(DesignSpec(), name) for name in _HELD_FIELDS}

        root = QVBoxLayout(self)
        root.setContentsMargins(SPACING["md"], SPACING["md"],
                                SPACING["md"], SPACING["md"])
        root.setSpacing(SPACING["sm"])

        header = ModuleHeader(
            APP_NAME,
            description=APP_DESCRIPTION,
            instruction="Describe the library and the effect you want to "
                        "catch; the curves say how many wells.",
        )
        self._header = header
        root.addWidget(header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_form())
        splitter.addWidget(self._build_output())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

        self._status = QLabel("")
        self._status.setObjectName(STATUS_OBJECT)
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        self._sync_derived()
        self._update_controls()

    # -- construction ------------------------------------------------------

    def _build_form(self) -> QWidget:
        holder = QScrollArea()
        holder.setWidgetResizable(True)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(0, 0, SPACING["sm"], 0)
        layout.setSpacing(SPACING["sm"])
        defaults = DesignSpec()

        library = QGroupBox("Library")
        form = QFormLayout(library)
        self._genes = self._int_box(2, 100000, defaults.n_genes)
        self._genes.setToolTip(
            "Genes in the library. The real T. gondii screen this simulator "
            "was fitted to had 452. Power falls roughly as the log of this.")
        self._grnas = self._int_box(1, 100, defaults.n_grnas_per_gene)
        self._grnas.setToolTip(
            "Guides per gene. Only reaches the simulation when you score per "
            "guide — see the caveats: there is no guide-efficiency layer.")
        self._score_per = QComboBox()
        self._score_per.addItems(["gene", "guide"])
        self._score_per.setToolTip(
            "'gene' pools a gene's guides before the model sees them, which "
            "is what the real analysis and spaCRPower do. 'guide' gives every "
            "construct its own coefficient and its own read count.")
        self._constructs = self._float_box(0.1, 500.0, defaults.constructs_per_well,
                                           decimals=2, step=0.1)
        self._constructs.setToolTip(
            "Mean library constructs spotted into each well — 'gRNAs per "
            "well'. The knob that trades constructs-per-well against "
            "wells-per-construct, and the sweep spaCRPower cared most about. "
            "4.6 in the real screen.")
        form.addRow("Genes", self._genes)
        form.addRow("gRNAs / gene", self._grnas)
        form.addRow("Score per", self._score_per)
        form.addRow("Constructs / well", self._constructs)
        self._library_note = QLabel("")
        self._library_note.setWordWrap(True)
        form.addRow(self._library_note)
        layout.addWidget(library)

        plate = QGroupBox("Plates")
        form = QFormLayout(plate)
        self._plate_format = QComboBox()
        for fmt in PLATE_FORMATS:
            self._plate_format.addItem(str(fmt), fmt)
        self._plate_format.setCurrentText(str(defaults.wells_per_plate))
        self._plates = self._int_box(1, 200, defaults.n_plates)
        self._plates.setToolTip("Plates in the screen. 4 x 384 in the real screen.")
        form.addRow("Wells / plate", self._plate_format)
        form.addRow("Plates", self._plates)
        self._wells_note = QLabel("")
        form.addRow(self._wells_note)
        layout.addWidget(plate)

        effect = QGroupBox("Effect")
        form = QFormLayout(effect)
        self._background = self._float_box(
            0.0001, 0.99, defaults.background_positive_rate, decimals=4, step=0.01)
        self._background.setToolTip(
            "Probability a non-hit cell is called positive — the classifier's "
            "false-positive rate. 0.12 in the real screen.")
        # The floor is below 1 on purpose. A spin box that refuses the
        # keystroke teaches nothing; one that accepts a protective effect and
        # then says why the model cannot score it teaches the thing worth
        # knowing — see DesignSpec.validate.
        self._effect = self._float_box(0.05, 50.0, defaults.effect_fold,
                                       decimals=3, step=0.1)
        self._effect.setToolTip(
            "How many times more often a hit-genotype cell is called "
            "positive. The effect size. The real screen's classifier sat at "
            "0.80 against a background of 0.12, i.e. 6.67-fold. Below 1 means "
            "a protective knockout, which this model does not score — it "
            "ranks evidence in one direction only.")
        self._prevalence = self._float_box(0.0001, 1.0, defaults.hit_rate,
                                           decimals=4, step=0.005)
        self._prevalence.setToolTip(
            "Fraction of the library that is a true hit. 0.025 was inferred "
            "from the real screen by inverting the well positivity rate "
            "against the classifier operating point. The single number most "
            "worth checking against your own pilot data.")
        form.addRow("Background positive rate", self._background)
        form.addRow("Effect size (fold)", self._effect)
        form.addRow("Hit prevalence", self._prevalence)
        self._effect_note = QLabel("")
        self._effect_note.setWordWrap(True)
        form.addRow(self._effect_note)
        layout.addWidget(effect)

        acquisition = QGroupBox("Imaging and sequencing")
        form = QFormLayout(acquisition)
        self._cells = self._float_box(1.0, 100000.0, defaults.cells_per_well,
                                      decimals=1, step=10.0)
        self._cells.setToolTip(
            "Mean cells imaged per well — the parameter you buy with "
            "microscope time, and the one the first curve sweeps. The real "
            "screen averaged 123.")
        self._reads = self._int_box(100, 10_000_000, int(defaults.reads_per_well))
        self._reads.setToolTip(
            "Mean sequencing reads per well. Unambiguously per well: "
            "spaCRPower divided its read budget by the number of genes. "
            "~30 000 in the real screen.")
        form.addRow("Cells imaged / well", self._cells)
        form.addRow("Reads / well", self._reads)
        # Everything the simulator needs that the form does not ask for is
        # printed rather than left implicit: a power analysis defended in a
        # methods section needs every number that went into it, and a
        # parameter that only exists in a dataclass default is a parameter
        # nobody knows they accepted.
        self._held_note = QLabel("")
        self._held_note.setWordWrap(True)
        self._held_note.setToolTip(
            "The simulator parameters this screen does not ask for, held at "
            "the values fitted to the real T. gondii screen. Change them by "
            "constructing a DesignSpec and calling set_spec().")
        form.addRow(self._held_note)
        layout.addWidget(acquisition)

        run = QGroupBox("Run")
        form = QFormLayout(run)
        self._replicates = self._int_box(1, 50, defaults.n_replicates)
        self._replicates.setToolTip(
            "Simulated screens per grid point. One screen at one setting is a "
            "single draw from a noisy process; three is the minimum that "
            "reads as a probability at all.")
        self._threshold = self._float_box(0.5, 1.0, defaults.detection_auroc,
                                          decimals=2, step=0.01)
        self._threshold.setToolTip(
            "The AUROC a simulated screen has to reach to count as a "
            "detection. There is no p-value here — the model ranks genes, so "
            "the bar is a ranking quality, and you choose it.")
        self._seed = self._int_box(0, 2_000_000_000, defaults.seed)
        self._seed.setToolTip(
            "Master seed. Every number on this screen is reproducible from "
            "this plus the parameters above.")
        self._backend = QComboBox()
        self._backend.addItems(list(_BACKEND_CHOICES))
        self._backend.setCurrentText(defaults.backend)
        self._backend.setToolTip(
            "Inference backend. 'torch' is mean-field ADVI and is always "
            "available; numpyro and pymc are exact NUTS if you installed "
            "them. 'auto' prefers NUTS and reports which it used.")
        form.addRow("Replicates / point", self._replicates)
        form.addRow("Detect at AUROC ≥", self._threshold)
        form.addRow("Seed", self._seed)
        form.addRow("Backend", self._backend)
        self._cost_note = QLabel("")
        self._cost_note.setWordWrap(True)
        form.addRow(self._cost_note)
        layout.addWidget(run)

        buttons = QHBoxLayout()
        self._btn_run = QPushButton("Run the power analysis")
        self._btn_run.clicked.connect(self.run)
        self._btn_stop = QPushButton("Stop")
        self._btn_stop.clicked.connect(self.cancel)
        buttons.addWidget(self._btn_run)
        buttons.addWidget(self._btn_stop)
        layout.addLayout(buttons)
        layout.addStretch(1)

        for widget in (self._genes, self._grnas, self._plates, self._reads,
                       self._seed, self._replicates):
            widget.valueChanged.connect(self._sync_derived)
        for widget in (self._constructs, self._background, self._effect,
                       self._prevalence, self._cells, self._threshold):
            widget.valueChanged.connect(self._sync_derived)
        self._score_per.currentTextChanged.connect(self._sync_derived)
        self._plate_format.currentTextChanged.connect(self._sync_derived)

        holder.setWidget(inner)
        holder.setMinimumWidth(320)
        return holder

    def _build_output(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING["sm"])

        self._answer = QLabel("No run yet — set the design and press Run.")
        self._answer.setObjectName(ANSWER_OBJECT)
        self._answer.setWordWrap(True)
        self._answer.setTextInteractionFlags(Qt.TextSelectableByMouse)
        font = self._answer.font()
        font.setPointSizeF(font.pointSizeF() * 1.15)
        self._answer.setFont(font)
        layout.addWidget(self._answer)

        self._caveats = CaveatPanel()
        layout.addWidget(self._caveats)

        self._cells_view = PowerCurveView("Detection probability vs cells per well")
        self._wells_view = PowerCurveView("Detection probability vs wells")
        layout.addWidget(self._cells_view, 1)
        layout.addWidget(self._wells_view, 1)

        self._table = QTableWidget(0, len(_TABLE_HEADERS))
        self._table.setHorizontalHeaderLabels(list(_TABLE_HEADERS))
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.NoSelection)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self._table.setMinimumHeight(140)
        # The caveat list and the sample-size table are the two regions
        # of this column with nothing behind them; the two curve views
        # between them paint their own panel in `paintEvent`.
        mark_surface(self._caveats, self._table)
        layout.addWidget(self._table, 1)
        return panel

    @staticmethod
    def _int_box(low: int, high: int, value: int) -> QSpinBox:
        box = QSpinBox()
        box.setRange(int(low), int(high))
        box.setValue(int(value))
        return box

    @staticmethod
    def _float_box(low: float, high: float, value: float, *,
                   decimals: int = 2, step: float = 0.1) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setDecimals(int(decimals))
        box.setRange(float(low), float(high))
        box.setSingleStep(float(step))
        box.setValue(float(value))
        return box

    # -- the design --------------------------------------------------------

    def spec(self) -> DesignSpec:
        """The design currently on the form.

        :returns: a :class:`~spacr.qt.widgets.power_design.DesignSpec`. This
            is the only thing the sweep is given, which is what makes an
            exported run re-runnable from the record.
        """
        return DesignSpec(
            n_genes=int(self._genes.value()),
            n_grnas_per_gene=int(self._grnas.value()),
            score_per=self._score_per.currentText(),
            cells_per_well=float(self._cells.value()),
            wells_per_plate=int(self._plate_format.currentData()
                                or self._plate_format.currentText()),
            n_plates=int(self._plates.value()),
            constructs_per_well=float(self._constructs.value()),
            background_positive_rate=float(self._background.value()),
            effect_fold=float(self._effect.value()),
            hit_rate=float(self._prevalence.value()),
            reads_per_well=float(self._reads.value()),
            n_replicates=int(self._replicates.value()),
            detection_auroc=float(self._threshold.value()),
            seed=int(self._seed.value()),
            backend=self._backend.currentText(),
            **self._held,
        )

    def set_spec(self, spec: DesignSpec) -> None:
        """Put ``spec`` on the form. For tests and for reloading a run.

        The fields the form does not show are carried in :attr:`_held`, so a
        spec round-trips: ``screen.set_spec(s); screen.spec() == s``. Dropping
        them would quietly re-simulate a different screen from the one asked
        for, with no visible difference on the form.
        """
        self._held = {name: getattr(spec, name) for name in _HELD_FIELDS}
        self._genes.setValue(int(spec.n_genes))
        self._grnas.setValue(int(spec.n_grnas_per_gene))
        self._score_per.setCurrentText(str(spec.score_per))
        self._cells.setValue(float(spec.cells_per_well))
        self._plate_format.setCurrentText(str(spec.wells_per_plate))
        self._plates.setValue(int(spec.n_plates))
        self._constructs.setValue(float(spec.constructs_per_well))
        self._background.setValue(float(spec.background_positive_rate))
        self._effect.setValue(float(spec.effect_fold))
        self._prevalence.setValue(float(spec.hit_rate))
        self._reads.setValue(int(spec.reads_per_well))
        self._replicates.setValue(int(spec.n_replicates))
        self._threshold.setValue(float(spec.detection_auroc))
        self._seed.setValue(int(spec.seed))
        self._backend.setCurrentText(str(spec.backend))
        self._sync_derived()

    def _sync_derived(self, *_args) -> None:
        """Recompute every derived label. Cheap, so it runs on each edit."""
        spec = self.spec()
        units = spec.n_library_units
        if spec.score_per == "guide":
            self._library_note.setText(
                f"{units} constructs get their own coefficient "
                f"({spec.n_genes} genes x {spec.n_grnas_per_gene} guides). "
                "Every guide of a hit gene is itself a hit — this prices the "
                "dilution of a bigger library, not the insurance of several "
                "guides.")
        else:
            self._library_note.setText(
                f"{units} genes get a coefficient; the "
                f"{spec.n_grnas_per_gene} guides are pooled before the model "
                "sees them, exactly as spaCRPower and the real analysis do.")
        self._wells_note.setText(f"{spec.n_wells} wells in the screen.")
        self._held_note.setText(
            "Held fixed at the real screen's fitted values: "
            + ", ".join(f"{name}={getattr(spec, name)!s}"
                        for name in _HELD_FIELDS) + ".")
        self._effect_note.setText(
            f"Hit-genotype cells are called positive "
            f"{spec.hit_positive_rate:.3f} of the time against a background "
            f"of {spec.background_positive_rate:.3f}; "
            f"{spec.expected_hits:.1f} of the {units} library units are hits.")
        seconds = estimate_runtime_s(spec)
        self._cost_note.setText(
            f"{(len(cells_grid(spec)) + len(wells_grid(spec))) * spec.n_replicates}"
            f" fits — very roughly {self._humanise(seconds)}. Rough: the "
            "estimate scales one measured fit by design size.")
        self._update_controls()

    @staticmethod
    def _humanise(seconds: float) -> str:
        if seconds < 90:
            return f"{seconds:.0f} s"
        if seconds < 5400:
            return f"{seconds / 60:.0f} min"
        return f"{seconds / 3600:.1f} h"

    # -- running -----------------------------------------------------------

    def run(self) -> bool:
        """Start both sweeps. Returns whether one was started.

        :returns: ``True`` if a sweep started (or, when ``threaded=False``,
            ran to completion successfully).
        """
        if self._busy:
            return False
        spec = self.spec()
        problems = spec.validate()
        if problems:
            self._set_status(problems[0], error=True)
            return False

        self._cancel = threading.Event()
        payload: Dict[str, Any] = {
            "spec": spec,
            "cancel": self._cancel,
            "progress": (self._worker_progress if self._threaded
                         else self._inline_progress),
            "fit_kwargs": dict(self.fit_kwargs),
        }
        self._set_status(
            f"Running {(len(cells_grid(spec)) + len(wells_grid(spec))) * spec.n_replicates}"
            f" fits on the {spec.backend} backend…")

        if not self._threaded:
            ok = True
            try:
                run_power_sweep(payload)
                self._apply_result(payload.get("result"))
            except Exception as exc:  # noqa: BLE001 - reported inline
                LOG.info("power sweep failed", exc_info=True)
                self._on_job_error(exc)
                ok = False
            self._busy = False
            self._update_controls()
            self.job_finished.emit(ok)
            return ok

        from ..bridge import make_thread

        # journal=False: the sweep reads no user data and writes no files, so
        # there is no artefact for a reproducibility manifest to describe —
        # and a housekeeping job that blocks shutdown is what
        # `RunRegistry.cancel_all` documents as the way to hang a headless run.
        # The record that matters (seed, backend, every parameter) is the
        # DesignSpec, which is on screen.
        thread, worker = make_thread(run_power_sweep, payload,
                                     app_key=APP_KEY, journal=False)
        self._jobs.append((thread, worker))
        self._pending.append(payload)
        worker.error.connect(self._on_worker_error_text)
        # Bound QWidget methods: Qt queues these back onto the GUI thread. A
        # closure would run on PipelineWorker's thread and must never touch a
        # label or a table.
        worker.line_ready.connect(self._on_worker_line)
        worker.finished.connect(self._on_job_settled)
        thread.finished.connect(self._retire_finished_jobs)
        self._busy = True
        self._update_controls()
        thread.start()
        return True

    def cancel(self) -> None:
        """Ask the running sweep to stop after the fit in flight.

        The fit itself is atomic — there is no safe point inside an ADVI
        optimisation to abandon — so this sets the event the sweep's
        ``on_point`` hook checks between grid points, and asks the worker to
        cancel as well so the run registry and Home see it too.
        """
        if self._cancel is not None:
            self._cancel.set()
        for _thread, worker in list(self._jobs):
            try:
                worker.request_cancel("cancelled from the Power screen")
            except (RuntimeError, AttributeError):
                pass
        if self._busy:
            self._set_status("Stopping after the fit in flight…")

    def _worker_progress(self, done: int, total: int, label: str) -> None:
        """Progress, called ON THE WORKER THREAD. Prints; emits nothing.

        :class:`~spacr.qt.bridge.PipelineWorker` redirects the job's stdout
        and re-emits it as ``line_ready``, which is the one mechanism the
        whole Qt layer uses to get text out of a worker — and the only one
        that is unambiguously safe, since a signal emitted from here and
        connected somewhere with a DirectConnection would touch widgets off
        the GUI thread, which has already aborted this process once (see the
        removed idle-flush pump in ``bridge.PipelineWorker.run``).

        The ``Progress: n/total`` spelling is what ``bridge._PROGRESS_RE``
        parses, so this run's bar appears on the Home screen without
        teaching Home anything about power analysis.
        """
        print(f"Progress: {int(done)}/{int(total)} ({label})", flush=True)

    def _inline_progress(self, done: int, total: int, label: str) -> None:
        """Progress for ``threaded=False``, where there is only one thread."""
        self.progressed.emit(int(done), int(total))

    def _on_worker_line(self, chunk: str) -> None:
        """GUI thread: turn a worker's progress line back into a signal."""
        for line in str(chunk).splitlines():
            match = _PROGRESS_LINE.search(line)
            if match:
                self.progressed.emit(int(match.group(1)), int(match.group(2)))

    # -- results -----------------------------------------------------------

    def result(self) -> Optional[Dict[str, Any]]:
        """The last sweep's result dict, or ``None``."""
        return self._result

    def _apply_result(self, result: Optional[Dict[str, Any]]) -> None:
        """Render a finished sweep. GUI thread only.

        Renders against the spec the sweep was RUN with, not the one on the
        form: a sweep is minutes long, the form is editable throughout, and
        labelling the result with a design that was never simulated is the
        easiest way for this screen to lie.
        """
        self._result = result
        if not result:
            self._set_status("The sweep produced no result.", error=True)
            return
        spec = result.get("spec") or self.spec()
        cells = result.get("cells_curve")
        wells = result.get("wells_curve")

        self._answer.setText(plain_sentence(spec, cells, wells))
        self._cells_view.set_curve(cells, "cells imaged per well",
                                   marker=float(round(spec.cells_per_well)),
                                   threshold=spec.detection_auroc)
        self._wells_view.set_curve(wells, "wells in the screen",
                                   marker=float(spec.n_wells),
                                   threshold=spec.detection_auroc)
        self._fill_table(cells, wells, spec)

        notes: List[str] = []
        if result.get("cancelled"):
            notes.append(
                "Stopped early — the curves show only the points that "
                "finished, not a design that ran out of power.")
        clipped = int(result.get("n_clipped_screens", 0) or 0)
        if clipped:
            notes.append(
                f"{clipped} simulated screen(s) clipped a gene-in-well "
                f"probability above 1, so the realised constructs per well "
                f"is below the {spec.constructs_per_well:g} you asked for. "
                "Raise the library evenness or lower constructs per well.")
        withheld = 0
        for curve in (cells, wells):
            if curve is not None and len(curve):
                withheld += int(curve["n_not_converged"].sum())
                withheld += int(curve["n_failed"].sum())
        if withheld:
            notes.append(
                f"{withheld} replicate(s) produced no usable fit and are "
                "counted as non-detections.")
        self._set_status(" ".join(notes) if notes else "Done.")

    def _fill_table(self, cells, wells, spec: DesignSpec) -> None:
        rows: List[Tuple[str, str, Any]] = []
        if cells is not None:
            for _, row in cells.iterrows():
                rows.append((f"{row['value']:g}", f"{spec.n_wells}", row))
        if wells is not None:
            for _, row in wells.iterrows():
                rows.append((f"{spec.cells_per_well:g}",
                             f"{row['value']:g}", row))
        self._table.setRowCount(len(rows))
        for index, (cell_text, well_text, row) in enumerate(rows):
            values = [
                cell_text,
                well_text,
                f"{100.0 * float(row['power']):.0f}%",
                f"{int(row['n_detected'])}/{int(row['n_replicates'])}",
                self._fmt(row["mean_auroc"]),
                self._fmt(row["mean_ap"]),
                self._fmt(row["ap_baseline"], places=3),
                str(int(row["n_not_converged"])),
                str(int(row["n_failed"])),
            ]
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self._table.setItem(index, column, item)

    @staticmethod
    def _fmt(value, places: int = 2) -> str:
        """A metric as text; a withheld metric as an em dash, never as 0.5."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "—"
        return "—" if not math.isfinite(number) else f"{number:.{places}f}"

    def table_rows(self) -> List[List[str]]:
        """Every table cell as plain strings. For tests."""
        return [[(self._table.item(r, c).text()
                  if self._table.item(r, c) else "")
                 for c in range(self._table.columnCount())]
                for r in range(self._table.rowCount())]

    def answer_text(self) -> str:
        """The headline sentence currently on screen."""
        return self._answer.text()

    def caveat_text(self) -> str:
        """Every caveat line on screen, shown or hidden."""
        return self._caveats.all_text()

    def visible_caveat_text(self) -> str:
        """Only the caveat lines the user can see without clicking."""
        return self._caveats.visible_text()

    def status_text(self) -> str:
        """The inline status line."""
        return self._status.text()

    # -- job plumbing ------------------------------------------------------

    def _on_job_settled(self, ok: bool) -> None:
        """Apply the finished sweep on the GUI thread.

        A partial result is rendered whether or not the worker reports
        success. Cancelling mid-sweep leaves real, finished grid points in
        the payload, and throwing them away because the run as a whole did
        not complete would make Stop destructive — the user pressed it
        because they had seen enough, not because they wanted the answer
        deleted. The curves say they are partial; see :meth:`_apply_result`.
        """
        self._busy = False
        payload = self._pending.pop(0) if self._pending else {}
        result = payload.get("result")
        ok = bool(ok)
        if result is not None:
            try:
                self._apply_result(result)
            except Exception as exc:  # noqa: BLE001 - reported inline
                LOG.info("could not render the sweep", exc_info=True)
                self._on_job_error(exc)
                ok = False
        elif self._cancel is not None and self._cancel.is_set():
            self._set_status("Stopped before the first fit finished.")
        elif ok:
            self._set_status("The sweep produced no result.", error=True)
        self._update_controls()
        self.job_finished.emit(ok and result is not None)

    def _retire_finished_jobs(self) -> None:
        """Retire every job whose QThread has stopped. GUI thread only.

        A BOUND METHOD, not a closure — the rule ``make_thread``'s own
        docstring states and then relies on for ``handle.retire``. With a
        closure PySide6 makes the QThread itself the receiver, and
        ``make_thread`` connects ``thread.finished -> thread.deleteLater``
        FIRST; slots run in connection order, so the DeferredDelete is posted
        ahead of the closure's metacall and Qt discards queued events for a
        destroyed receiver. The job is then never retired, ``active_jobs()``
        never returns to zero, and every ``waitUntil(active_jobs() == 0)``
        sits there until it times out with the QThread's C++ half already
        gone.

        It sweeps rather than naming a sender for the same reason: by the
        time this runs the emitter may be exactly what is gone, and
        ``QObject.sender()`` is null for a queued call whose emitter was
        destroyed.
        """
        from ..bridge import thread_has_stopped

        for thread, _worker in list(self._jobs):
            if thread_has_stopped(thread):
                self._jobs = [(t, w) for (t, w) in self._jobs if t is not thread]

    def active_jobs(self) -> int:
        """How many sweep threads are still winding down."""
        return len(self._jobs)

    def is_busy(self) -> bool:
        """Whether a sweep is in flight."""
        return self._busy

    def _on_worker_error_text(self, tb: str) -> None:
        """Turn a worker traceback into one inline line (never a dialog)."""
        line = ""
        for candidate in reversed(str(tb).strip().splitlines()):
            if candidate.strip():
                line = candidate.strip()
                break
        self._set_status(f"The sweep failed: {line or 'unknown error'}",
                         error=True)

    def _on_job_error(self, exc: Exception) -> None:
        message = str(exc) or exc.__class__.__name__
        self._set_status(f"The sweep failed: {message}", error=True)

    def _set_status(self, text: str, error: bool = False) -> None:
        """Put one line under the form. Never a dialog — a modal hangs headless.

        The colour is a dynamic property re-polished in place rather than an
        inline stylesheet, so it comes from the live palette and survives a
        theme switch. See :func:`_power_qss`.
        """
        self._status.setProperty("spacrError", "true" if error else "false")
        style = self._status.style()
        if style is not None:
            style.unpolish(self._status)
            style.polish(self._status)
        self._status.setText(str(text))

    def status_is_error(self) -> bool:
        """Whether the status line is currently showing a failure. For tests."""
        return self._status.property("spacrError") == "true"

    def _update_controls(self) -> None:
        problems = self.spec().validate() if hasattr(self, "_genes") else []
        self._btn_run.setEnabled(not self._busy and not problems)
        self._btn_stop.setEnabled(self._busy)
        if problems and not self._busy:
            self._btn_run.setToolTip(problems[0])
        else:
            self._btn_run.setToolTip(
                "Simulate the screen at each point on both sweeps, fit the "
                "model to each, and report how often it found the hits.")

    # -- shutdown ----------------------------------------------------------

    def closeEvent(self, event):  # noqa: N802 - Qt name
        """Let every in-flight sweep stop before the widget dies."""
        self.cancel()
        for thread, _worker in list(self._jobs):
            try:
                if thread.isRunning():
                    thread.quit()
                    thread.wait(10000)
            except RuntimeError:
                pass
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def make_power_screen(app_key: Optional[str] = None) -> QWidget:
    """Factory handed to :func:`spacr.qt.app.register_app`."""
    return PowerScreen()


APP_NAME = "Power / Design"
APP_DESCRIPTION = ("How many cells per well and how many wells to detect an "
                   "effect of a given size")
APP_INTRO = (
    "Before a pooled screen runs, the only honest way to know whether it can "
    "find its hits is to simulate screens you know the truth for and fit the "
    "model you would really use. Describe the library, the plates, the "
    "classifier and the effect you expect; this sweeps cells-per-well and "
    "wells, and reports the fraction of simulated screens in which the model "
    "recovered the planted hits. The departures from the R package it is "
    "ported from — including that the R version overstates power — are shown "
    "next to the number, not in a footnote.")
#: What `spacr.cli.INTERACTIVE_ONLY` wants: why this app has no headless run.
APP_CLI_NOTE = ("Interactive design exploration; "
                "spacr.power_model.scan_parameters() is the headless "
                "equivalent and takes the same parameters.")
#: :data:`APP_NAME` in the nine non-English UI languages, in
#: :data:`spacr.qt.i18n.LANGUAGES` order after English — sv, de, es, zh_CN,
#: pt, hi, ko, is, fr. Handed to ``register_app(translations=…)``, which is
#: what puts them in every catalog; a missing one is a blank sidebar row in
#: that language rather than an English one. "Power" is the statistical
#: term throughout, not electrical power.
APP_TRANSLATIONS = (
    "Styrka / design",
    "Teststärke / Design",
    "Potencia / diseño",
    "检验效能 / 设计",
    "Poder / delineamento",
    "सांख्यिकीय शक्ति / डिज़ाइन",
    "검정력 / 설계",
    "Tölfræðilegt afl / hönnun",
    "Puissance / plan",
)


#: The settings this app has, as ``{key: (default, type, tooltip)}``.
#:
#: The screen's own form IS its settings — every one of these is a spin box
#: on it — but the key still needs a record in :mod:`spacr.settings` for the
#: generic machinery (``spacr-run power``, the settings diff, the
#: per-app inventory tests) to see anything at all. Prefixed ``power_`` so
#: none of the ~800 existing keys collide.
_SETTINGS: Dict[str, Tuple[Any, Any, str]] = {
    "power_n_genes": (
        452, int,
        "(int) - Genes in the library. The real T. gondii screen this "
        "simulator was fitted to had 452. Power falls roughly as the log of "
        "this: doubling the library costs about as much as halving the wells."),
    "power_n_grnas_per_gene": (
        4, int,
        "(int) - Guides per gene. Only reaches the simulation when "
        "power_score_per is 'guide'; there is no guide-efficiency layer in "
        "the port, so scoring per gene it changes no number."),
    "power_score_per": (
        "gene", str,
        "(str) - 'gene' pools a gene's guides before the model sees them, "
        "which is what the real analysis and spaCRPower do; 'guide' gives "
        "every construct its own coefficient."),
    "power_cells_per_well": (
        123.0, float,
        "(float) - Mean cells imaged per well. The real screen averaged 123. "
        "This is the parameter you buy with microscope time, and the first "
        "curve sweeps it."),
    "power_wells_per_plate": (
        384, int,
        "(int) - Wells per plate: 96, 384 or 1536. With the plate count it "
        "sets the total number of wells, and raising it buys replication "
        "rather than cells - a 1536 plate holds four times the wells of a 384 "
        "at the same imaging cost per well, so power rises while cells per "
        "well stays where you set it."),
    "power_n_plates": (
        4, int,
        "(int) - Plates in the screen; four 384-well plates in the real one. "
        "This is the cheapest axis to move: another plate is another full set "
        "of wells, and because plate is a random effect in the model it also "
        "buys a better estimate of plate-to-plate variance rather than "
        "confounding with it."),
    "power_constructs_per_well": (
        4.6, float,
        "(float) - Mean library constructs spotted into each well. The knob "
        "that trades constructs-per-well against wells-per-construct, and "
        "the sweep spaCRPower cared most about. 4.6 in the real screen."),
    "power_background_positive_rate": (
        0.12, float,
        "(float) - Probability a non-hit cell is called positive - the "
        "classifier's false-positive rate. 0.12 in the real screen."),
    "power_effect_fold": (
        6.667, float,
        "(float) - How many times more often a hit-genotype cell is called "
        "positive. The effect size. The real screen's MaxViT classifier sat "
        "at 0.80 against 0.12, i.e. 6.67-fold."),
    "power_hit_rate": (
        0.025, float,
        "(float) - Fraction of the library that is a true hit. 0.025 was "
        "inferred from the real screen by inverting the well positivity rate "
        "against the classifier operating point. The single number most "
        "worth checking against your own pilot data."),
    "power_reads_per_well": (
        30000, int,
        "(int) - Mean sequencing reads per well. Unambiguously per well: "
        "spaCRPower divided its read budget by the number of genes, giving "
        "~284 where a real screen has ~30000."),
    "power_n_replicates": (
        3, int,
        "(int) - Simulated screens per grid point. One screen at one setting "
        "is a single draw from a noisy process."),
    "power_detection_auroc": (
        0.80, float,
        "(float) - The AUROC a simulated screen must reach to count as a "
        "detection. There is no p-value here - the model ranks genes, so the "
        "bar is a ranking quality and you choose it."),
    "power_seed": (
        0, int,
        "(int) - Master seed. Every number the screen reports is reproducible "
        "from this plus the parameters above."),
    "power_backend": (
        "torch", str,
        "(str) - Inference backend: 'auto', 'torch' (mean-field ADVI, always "
        "available), 'numpyro' or 'pymc' (exact NUTS, if installed). ADVI "
        "gets the coefficient ORDER right, which is all AUROC needs, and "
        "understates the intervals, which is why it is not the choice when "
        "the interval is the deliverable."),
}


def power_default_settings(settings: Optional[Dict[str, Any]] = None
                           ) -> Dict[str, Any]:
    """The app's default settings dict, in :mod:`spacr.settings` shape.

    :param settings: existing settings to fill in; a fresh dict if omitted.
    :returns: ``settings``, with every missing ``power_`` key defaulted.
    """
    settings = dict(settings or {})
    for key, (default, _type, _tip) in _SETTINGS.items():
        settings.setdefault(key, default)
    return settings


def spec_from_settings(settings: Dict[str, Any]) -> DesignSpec:
    """Build a :class:`DesignSpec` from a settings dict.

    The bridge between the generic settings machinery and this screen, so a
    design saved as settings and a design typed into the form are the same
    object by the time either reaches the simulator.

    :param settings: any mapping; missing keys take their defaults.
    :returns: the design.
    """
    filled = power_default_settings(settings)
    return DesignSpec(
        n_genes=int(filled["power_n_genes"]),
        n_grnas_per_gene=int(filled["power_n_grnas_per_gene"]),
        score_per=str(filled["power_score_per"]),
        cells_per_well=float(filled["power_cells_per_well"]),
        wells_per_plate=int(filled["power_wells_per_plate"]),
        n_plates=int(filled["power_n_plates"]),
        constructs_per_well=float(filled["power_constructs_per_well"]),
        background_positive_rate=float(filled["power_background_positive_rate"]),
        effect_fold=float(filled["power_effect_fold"]),
        hit_rate=float(filled["power_hit_rate"]),
        reads_per_well=float(filled["power_reads_per_well"]),
        n_replicates=int(filled["power_n_replicates"]),
        detection_auroc=float(filled["power_detection_auroc"]),
        seed=int(filled["power_seed"]),
        backend=str(filled["power_backend"]),
    )


def register_settings(replace: bool = False) -> bool:
    """Register this app's defaults through :func:`spacr.settings.register_defaults`.

    Separate from :func:`register`, and like it **not called at import**:
    ``register_defaults`` merges into the process-wide ``expected_types``,
    ``tooltips`` and ``categories`` tables, and a module that mutates those
    the moment it is imported changes what every settings test in the suite
    sees depending on import order. ``register_app(defaults_module=...)``
    exists precisely so the import is deferred to the moment the settings
    panel asks for the key; this function is what that import would run.

    :param replace: overwrite an existing registration for this key.
    :returns: ``True`` if this call registered it, ``False`` if it was
        already there.
    """
    from ...settings import has_registered_defaults, register_defaults

    if has_registered_defaults(APP_KEY) and not replace:
        return False
    register_defaults(
        APP_KEY, power_default_settings, replace=replace,
        expected_types={k: v[1] for k, v in _SETTINGS.items()},
        tooltips={k: v[2] for k, v in _SETTINGS.items()},
        categories={"Power analysis": list(_SETTINGS)},
        description=APP_INTRO)
    return True


def register() -> bool:
    """Put Power / Design in the app registry, through the public seam.

    It claims :data:`spacr.qt.app.SECTION_DESIGN`, which is declared in
    ``SECTION_ORDER`` and has never had an app — its note already reads
    "Plan the experiment before it runs: power, sample size, plate layout,
    controls and replicates", so registering makes that tab appear with the
    description it was written for.

    :returns: ``True`` if this call is what registered it. Safe to call
        twice — a module imported from two paths must not raise.

    Called from ``app.py``'s ``_SELF_REGISTERING_APPS`` table, at the bottom
    of that module, which is the one point where a registration is visible to
    everybody and happens on ``import spacr.qt.app`` rather than only at
    launch. This module does not call it at its own import, so merely reading
    the screen's code does not add a row.

    **GUI-only, deliberately.** It passes ``cli_note`` and no ``entry``, so
    ``spacr-run power`` answers with :data:`APP_CLI_NOTE` instead of "unknown
    module". The reason is not that the sweep cannot run headless — it can,
    and the note names the call — but that its *inputs* are not a settings
    file. Every other ``spacr-run`` module takes a ``src`` and processes it;
    this one takes a design, and its output is a curve you read by comparing
    points on it. A settings.csv that pinned one point of that curve would be
    a worse interface to :func:`spacr.power_model.scan_parameters` than
    calling it, which is exactly what the note tells the user to do.
    """
    from ..app import APPS, SECTION_DESIGN, STAGE_ALPHA, register_app
    if any(row[0] == APP_KEY for row in APPS):
        return False
    # Settings first: if the shared tables reject a key, nothing has landed
    # in APPS yet and the app is absent rather than half-present.
    register_settings()
    register_app(
        APP_KEY, APP_NAME, APP_DESCRIPTION, SECTION_DESIGN,
        factory=make_power_screen, stage=STAGE_ALPHA,
        title="Power / Design", intro=APP_INTRO, cli_note=APP_CLI_NOTE,
        api_module="qt/screens/power",
        defaults_module="spacr.qt.screens.power",
        translations=APP_TRANSLATIONS)
    return True
