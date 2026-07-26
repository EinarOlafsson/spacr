"""Hyperparameter search panel — the Live-Preview-shaped window for sweeps.

Structurally this is the sibling of :mod:`spacr.qt.widgets.live_preview`: a
self-contained :class:`QWidget` that owns its own controls, runs the expensive
work on a :class:`QThread`, streams results back over signals, reports every
failure inline (never in a modal — a QMessageBox hangs a headless run), and
pushes a chosen configuration back into the host screen's settings panel through
a callback the host registers. Host screens embed it with
:func:`build_hyperparam_card`, exactly the way the Mask screen embeds the live
preview, and toggle it with a label reading "Hyperparameter search" where the
Mask screen's says "LP".

What it deliberately does *not* do is announce a winner and stop talking. The
table is ordered by the criterion the user picked, but the panel also draws the
small-multiples panel (embeddings for UMAP, score-versus-trial with the noise
band for the classifiers) and prints the spread, the within-noise flag and the
failure count, because those are the parts that say whether the winner means
anything. See :mod:`spacr.hyperparam` for why.

:author: spaCR
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QGridLayout, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QPushButton, QSizePolicy, QSpinBox, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ...hyperparam import (
    APP_CRITERIA, DEFAULT_SPACES, SearchResult, SearchSpace, Trial,
    run_search_for_app,
)

LOG = logging.getLogger("spacr.qt.hyperparam")

#: Label the host screens put on the toggle where the Mask screen says "LP".
TOGGLE_TEXT = "Hyperparameter search"

#: Tooltip for that toggle.
TOGGLE_TOOLTIP = (
    "Click to open the hyperparameter search. It sweeps the parameters you "
    "list, scores every configuration with a criterion you name, and reports "
    "the spread as well as the winner — so you can see whether the winner is "
    "real or noise. Nothing is applied to your settings until you press Apply."
)

#: Apps this panel can search, with the parameters it offers and their types.
#: ``(setting_key, label, kind)`` — kind drives the inline validation, so a
#: typo lands as a sentence under the controls instead of a traceback.
APP_PARAMS: Dict[str, Tuple[Tuple[str, str, str], ...]] = {
    "umap": (
        ("n_neighbors", "n_neighbors", "int"),
        ("min_dist", "min_dist", "float"),
        ("metric", "metric", "str"),
    ),
    "classify": (
        ("learning_rate", "learning_rate", "float"),
        ("dropout_rate", "dropout_rate", "float"),
        ("epochs", "epochs", "int"),
        ("weight_decay", "weight_decay", "float"),
    ),
    "ml_analyze": (
        ("learning_rate", "learning_rate", "float"),
        ("n_estimators", "n_estimators", "int"),
        ("reg_alpha", "reg_alpha", "float"),
        ("reg_lambda", "reg_lambda", "float"),
    ),
}

#: How many small multiples to draw. More than this and each panel is too small
#: to read, which defeats the point of looking at them.
MAX_PANELS = 12


# ---------------------------------------------------------------------------
# Pure helpers — no Qt, unit-testable without a display
# ---------------------------------------------------------------------------

def parse_values(text: str, kind: str, name: str) -> List[Any]:
    """Parse a comma-separated list of values for one parameter.

    :param text: the raw field contents, e.g. ``"5, 15, 50"``.
    :param kind: ``'int'``, ``'float'`` or ``'str'``.
    :param name: parameter name, used in the error message.
    :returns: the parsed values, in the order given.
    :raises ValueError: with a sentence the panel shows inline when a value
        does not match ``kind``.
    """
    parts = [p.strip() for p in (text or "").split(",")]
    parts = [p for p in parts if p]
    out: List[Any] = []
    for p in parts:
        if kind == "int":
            try:
                out.append(int(p))
            except ValueError:
                raise ValueError(
                    f"Parameter '{name}' takes whole numbers; {p!r} is not "
                    f"one. Write values like: 5, 15, 50"
                ) from None
        elif kind == "float":
            try:
                out.append(float(p))
            except ValueError:
                raise ValueError(
                    f"Parameter '{name}' takes numbers; {p!r} is not one. "
                    f"Write values like: 0.0, 0.1, 0.5"
                ) from None
        else:
            out.append(p)
    return out


def format_params(params: Dict[str, Any]) -> str:
    """Render a configuration as ``k=v, k=v`` in sorted-key order."""
    return ", ".join(f"{k}={params[k]}" for k in sorted(params))


def figure_to_pixmap(fig) -> QPixmap:
    """Rasterise a matplotlib figure into a QPixmap without touching disk.

    :param fig: the matplotlib Figure.
    :returns: the pixmap (null if rendering failed).
    """
    buf = BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=100,
                    facecolor=fig.get_facecolor())
    except Exception:
        LOG.debug("figure render failed", exc_info=True)
        return QPixmap()
    pm = QPixmap()
    pm.loadFromData(buf.getvalue(), "PNG")
    return pm


def build_panel_figure(result: SearchResult, max_panels: int = MAX_PANELS):
    """Draw the small-multiples panel for a finished (or partial) sweep.

    For a UMAP sweep this is one scatter per trial — the deliverable, because
    the scores cannot tell you an embedding is right. For a classifier sweep
    there is no embedding to draw, so it is score-versus-trial with the noise
    band shaded: every configuration inside the band is indistinguishable from
    the winner.

    :param result: the sweep to draw.
    :param max_panels: cap on the number of embedding panels.
    :returns: a matplotlib Figure, or None when there is nothing to draw.
    """
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    ranked = result.ranked()
    if not ranked:
        return None

    embedded = [t for t in ranked if t.extra_metrics.get("embedding") is not None]
    if embedded:
        shown = embedded[:max_panels]
        cols = min(4, len(shown))
        rows = (len(shown) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows),
                                 squeeze=False)
        for ax in axes.ravel():
            ax.set_axis_off()
        for i, trial in enumerate(shown):
            ax = axes[i // cols][i % cols]
            ax.set_axis_on()
            emb = trial.extra_metrics["embedding"]
            ax.scatter([p[0] for p in emb], [p[1] for p in emb],
                       s=4, alpha=0.7)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{format_params(trial.params)}\n"
                         f"{result.metric}={float(trial.score):.3f}",
                         fontsize=7)
        fig.suptitle(
            f"{len(shown)} of {len(ranked)} embeddings, ranked by "
            f"{result.metric} — look at them; the score is not a verdict",
            fontsize=8)
        fig.tight_layout()
        return fig

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    xs = list(range(1, len(ranked) + 1))
    ys = [float(t.score) for t in ranked]
    ax.plot(xs, ys, "o-", markersize=4)
    noise, source = result.noise_level()
    if noise:
        best = ys[0]
        lo = best - noise if result.higher_is_better else best
        hi = best if result.higher_is_better else best + noise
        ax.axhspan(min(lo, hi), max(lo, hi), alpha=0.18,
                   label=f"within noise ({source})")
        ax.legend(fontsize=7)
    ax.set_xlabel("rank")
    ax.set_ylabel(result.metric)
    ax.set_title(f"{len(ranked)} configurations by {result.metric}",
                 fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

@dataclass
class SearchRequest:
    """Everything the worker needs for one sweep.

    Kept as a plain dataclass so tests can construct one directly, exactly the
    way :class:`spacr.qt.widgets.live_preview.PreviewRequest` is used.
    """

    app_key: str = "umap"
    space: Optional[SearchSpace] = None
    settings: Dict[str, Any] = field(default_factory=dict)
    criterion: str = "trustworthiness"
    mode: str = "grid"
    n_trials: int = 12
    seed: int = 0
    n_folds: int = 5


class _SearchWorker(QThread):
    """Runs one sweep in the background, streaming trials as they complete."""

    trial_ready = Signal(object, int, int)   # (Trial, completed, total)
    search_done = Signal(object, str)        # (SearchResult or None, error)

    def __init__(self, request: SearchRequest, search_fn=None, parent=None):
        """Store the request and the (optionally injected) search function."""
        super().__init__(parent)
        self._request = request
        self._search_fn = search_fn
        self._stop = threading.Event()

    def request_stop(self) -> None:
        """Ask the sweep to stop after the trial currently in flight."""
        self._stop.set()

    @property
    def stopped(self) -> bool:
        """True once :meth:`request_stop` has been called."""
        return self._stop.is_set()

    def run(self):
        """Thread body: run the sweep, forwarding progress and failures."""
        req = self._request
        try:
            fn = self._search_fn or _default_search_fn
            result = fn(req,
                        lambda t, done, total: self.trial_ready.emit(
                            t, done, total),
                        self._stop.is_set)
            self.search_done.emit(result, "")
        except Exception as exc:
            LOG.info("hyperparameter search failed: %s", exc, exc_info=True)
            self.search_done.emit(None, f"{type(exc).__name__}: {exc}")


def _default_search_fn(request: SearchRequest, on_trial, should_stop) -> SearchResult:
    """Dispatch a request to :func:`spacr.hyperparam.run_search_for_app`."""
    return run_search_for_app(
        request.app_key, request.settings, request.space,
        criterion=request.criterion, mode=request.mode,
        n_trials=request.n_trials, seed=request.seed, n_folds=request.n_folds,
        on_trial=on_trial, should_stop=should_stop)


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class HyperparamPanel(QWidget):
    """Search-space controls, a live results table and a small-multiples panel.

    :param app_key: ``'umap'``, ``'classify'`` or ``'ml_analyze'``.
    :param parent: optional Qt parent.
    :ivar search_finished: emitted with the :class:`SearchResult` when a sweep
        ends (including a stopped, partial one).
    """

    search_finished = Signal(object)

    COLUMNS = ("#", "score", "fold sd", "parameters", "status")

    def __init__(self, app_key: str = "umap", parent=None):
        """Build the controls, table and preview for ``app_key``."""
        super().__init__(parent)
        if app_key not in APP_PARAMS:
            raise ValueError(
                f"No hyperparameter search is defined for {app_key!r}; "
                f"searchable apps are {sorted(APP_PARAMS)}.")
        self.app_key = app_key
        self._settings: Dict[str, Any] = {}
        self._worker: Optional[_SearchWorker] = None
        self._result: Optional[SearchResult] = None
        self._live_trials: List[Trial] = []
        self._search_fn = None
        self._apply_cb: Optional[Callable[[Dict[str, Any]], Any]] = None
        self._value_edits: Dict[str, QLineEdit] = {}
        self._build_ui()

    # -- construction ------------------------------------------------------

    def _build_ui(self) -> None:
        """Lay out controls on the left, table + preview on the right."""
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        # -- search space controls
        controls = QWidget()
        grid = QGridLayout(controls)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)
        defaults = DEFAULT_SPACES.get(self.app_key, {})
        for r, (key, label, kind) in enumerate(APP_PARAMS[self.app_key]):
            lab = QLabel(label)
            lab.setToolTip(
                f"Comma-separated {kind} values to try for '{key}'. Leave "
                f"empty to keep '{key}' out of the search.")
            edit = QLineEdit()
            edit.setPlaceholderText(f"comma-separated {kind} values")
            preset = defaults.get(key)
            if preset:
                edit.setText(", ".join(str(v) for v in preset))
            edit.setToolTip(lab.toolTip())
            grid.addWidget(lab, r, 0)
            grid.addWidget(edit, r, 1)
            self._value_edits[key] = edit
        root.addWidget(controls)

        # -- run row
        row = QHBoxLayout()
        row.setSpacing(6)

        row.addWidget(QLabel("criterion"))
        self._criterion = QComboBox()
        self._criterion.addItems(APP_CRITERIA[self.app_key])
        self._criterion.setToolTip(
            "Which named criterion ranks the trials. Different criteria reward "
            "different things and routinely pick different winners — that is "
            "not a bug, it is what 'best' means without a ground truth.")
        row.addWidget(self._criterion)

        row.addWidget(QLabel("mode"))
        self._mode = QComboBox()
        self._mode.addItems(["grid", "random"])
        self._mode.setToolTip(
            "grid evaluates every combination; random samples n trials from "
            "the space with the seed below, which is reproducible.")
        row.addWidget(self._mode)

        row.addWidget(QLabel("n trials"))
        self._n_trials = QSpinBox()
        self._n_trials.setRange(1, 10_000)
        self._n_trials.setValue(12)
        self._n_trials.setToolTip("How many configurations random mode draws.")
        row.addWidget(self._n_trials)

        row.addWidget(QLabel("folds"))
        self._n_folds = QSpinBox()
        self._n_folds.setRange(2, 50)
        self._n_folds.setValue(5)
        self._n_folds.setToolTip(
            "Cross-validation folds per trial. Folds are grouped by well, so "
            "crops from one well never straddle a split.")
        row.addWidget(self._n_folds)
        if self.app_key == "umap":
            self._n_folds.setVisible(False)

        row.addWidget(QLabel("seed"))
        self._seed = QSpinBox()
        self._seed.setRange(0, 1_000_000)
        self._seed.setValue(0)
        self._seed.setToolTip(
            "Fixes the sampling, the folds and the reducer, so the same seed "
            "reproduces the same sweep.")
        row.addWidget(self._seed)

        self._run_btn = QPushButton("Run search")
        self._run_btn.clicked.connect(self.run_search)
        row.addWidget(self._run_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.setToolTip(
            "Stop after the trial in flight. The trials already finished are "
            "kept and the result is marked partial.")
        self._stop_btn.clicked.connect(self.stop_search)
        row.addWidget(self._stop_btn)

        self._apply_btn = QPushButton("Apply configuration")
        self._apply_btn.setEnabled(False)
        self._apply_btn.setToolTip(
            "Write the selected row's parameters into the settings panel. "
            "Nothing changes until you press this.")
        self._apply_btn.clicked.connect(self.apply_selected)
        row.addWidget(self._apply_btn)

        row.addStretch(1)
        root.addLayout(row)

        # -- results table + preview
        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)

        self._table = QTableWidget(0, len(self.COLUMNS))
        self._table.setHorizontalHeaderLabels(list(self.COLUMNS))
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.Stretch)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        split.addWidget(self._table)

        self._preview = QLabel("No search has been run yet.")
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setMinimumWidth(220)
        self._preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._preview.setWordWrap(True)
        split.addWidget(self._preview)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 4)
        root.addWidget(split, 1)

        # -- status + caveats
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setObjectName("HyperparamStatus")
        root.addWidget(self._status)

        self._notes = QLabel("")
        self._notes.setWordWrap(True)
        self._notes.setObjectName("HyperparamNotes")
        root.addWidget(self._notes)

    # -- host integration --------------------------------------------------

    def set_apply_callback(self, cb: Optional[Callable[[Dict[str, Any]], Any]]) -> None:
        """Register the callback that writes a chosen config into the settings
        panel. Mirrors ``LivePreviewPanel.set_propagate_callback``.

        :param cb: called with the parameter dict when the user hits Apply.
        """
        self._apply_cb = cb

    def set_search_fn(self, fn) -> None:
        """Override the search backend.

        :param fn: ``fn(request, on_trial, should_stop) -> SearchResult``.
        """
        self._search_fn = fn

    def apply_settings(self, settings: Dict[str, Any]) -> None:
        """Adopt the host screen's current settings as the search's base.

        Any searched parameter that already has a value in ``settings`` and no
        list in its field is seeded with that single value, so a search always
        includes what the user has configured.

        :param settings: the host screen's settings dict.
        """
        self._settings = dict(settings or {})
        for key, _label, _kind in APP_PARAMS[self.app_key]:
            edit = self._value_edits[key]
            if edit.text().strip():
                continue
            value = self._settings.get(key)
            if value is not None:
                edit.setText(str(value))

    # -- search space ------------------------------------------------------

    def current_space(self) -> SearchSpace:
        """Build the :class:`SearchSpace` from the fields.

        :returns: the space.
        :raises ValueError: with a message meant to be shown inline when a
            field holds a value of the wrong type or nothing is filled in.
        """
        params: Dict[str, List[Any]] = {}
        for key, label, kind in APP_PARAMS[self.app_key]:
            values = parse_values(self._value_edits[key].text(), kind, label)
            if values:
                params[key] = values
        if not params:
            raise ValueError(
                "Nothing to search: fill in at least one parameter with at "
                "least one value, e.g. n_neighbors = 5, 15, 50.")
        return SearchSpace(params)

    # -- running -----------------------------------------------------------

    def run_search(self) -> bool:
        """Validate the space and start the sweep in the background.

        :returns: True when a sweep was started; False when validation failed
            (the reason is on the status label — never in a dialog).
        """
        if self._worker is not None and self._worker.isRunning():
            self._status.setText("A search is already running.")
            return False
        try:
            space = self.current_space()
        except ValueError as exc:
            self._status.setText(str(exc))
            return False

        request = SearchRequest(
            app_key=self.app_key,
            space=space,
            settings=dict(self._settings),
            criterion=self._criterion.currentText(),
            mode=self._mode.currentText(),
            n_trials=int(self._n_trials.value()),
            seed=int(self._seed.value()),
            n_folds=int(self._n_folds.value()),
        )
        self._result = None
        self._live_trials = []
        self._table.setRowCount(0)
        self._apply_btn.setEnabled(False)
        self._notes.setText("")
        self._preview.setText("Running…")
        self._preview.setPixmap(QPixmap())
        self._status.setText(
            f"Running {request.mode} search over {space.size()} "
            f"configurations, ranked by {request.criterion}…")

        worker = _SearchWorker(request, self._search_fn, self)
        worker.trial_ready.connect(self._on_trial_ready)
        worker.search_done.connect(self._on_search_done)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        worker.start()
        return True

    def stop_search(self) -> None:
        """Ask the running sweep to stop; the result comes back partial."""
        worker = self._worker
        if worker is None or not worker.isRunning():
            self._status.setText("No search is running.")
            return
        worker.request_stop()
        self._status.setText(
            "Stopping after the trial in flight — the finished trials are "
            "kept and the result is marked partial.")

    # -- signal handlers ---------------------------------------------------

    def _on_trial_ready(self, trial: Trial, done: int, total: int) -> None:
        """Append one finished trial to the table as the sweep progresses."""
        self._live_trials.append(trial)
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._set_row(row, str(trial.index + 1),
                      "-" if trial.score is None else f"{trial.score:.4f}",
                      self._fold_sd(trial),
                      format_params(trial.params),
                      "failed" if trial.error else "ok",
                      trial.params, trial.error)
        self._status.setText(
            f"{done} of {total} configurations evaluated"
            + (f" — last one failed: {trial.error}" if trial.error else ""))

    def _on_search_done(self, result: Optional[SearchResult], err: str) -> None:
        """Rebuild the table in ranked order and draw the preview."""
        self._worker = None
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        if err:
            self._status.setText(f"Search failed: {err}")
            self._preview.setText("Search failed.")
            return
        self._result = result
        self._rebuild_table(result)
        self._apply_btn.setEnabled(bool(result.ranked()))
        self._notes.setText("\n".join(f"• {n}" for n in result.notes))

        summary: List[str] = []
        if result.partial:
            summary.append(
                f"PARTIAL — stopped after {len(result.trials)} trials; this is "
                f"not a completed sweep.")
        if result.best is None:
            summary.append("No configuration produced a score.")
        else:
            stats = result.score_stats()
            summary.append(
                f"Best {result.metric}={float(result.best.score):.4f} at "
                f"{format_params(result.best.params)}; spread over "
                f"{stats['n']} trials {stats['worst']:.4f}…{stats['best']:.4f} "
                f"(sd {stats['std']:.4f}).")
            if result.within_noise():
                summary.append(
                    f"WITHIN NOISE — {len(result.trials_within_noise())} "
                    f"configurations are indistinguishable from the best; the "
                    f"winner is arbitrary.")
        if result.n_failed:
            summary.append(f"{result.n_failed} trials failed.")
        self._status.setText(" ".join(summary))
        self._draw_preview(result)
        self.search_finished.emit(result)

    # -- table -------------------------------------------------------------

    @staticmethod
    def _fold_sd(trial: Trial) -> str:
        """Render a trial's fold-to-fold standard deviation, or a dash."""
        sd = trial.extra_metrics.get("fold_std")
        try:
            return f"{float(sd):.4f}"
        except (TypeError, ValueError):
            return "-"

    def _set_row(self, row: int, rank: str, score: str, sd: str, params: str,
                 status: str, param_dict: Dict[str, Any],
                 error: Optional[str]) -> None:
        """Write one table row and stash the config on the first cell."""
        cells = (rank, score, sd, params, error or status)
        for col, text in enumerate(cells):
            item = QTableWidgetItem(text)
            if col == 0:
                item.setData(Qt.UserRole, dict(param_dict))
            if error:
                item.setToolTip(error)
            self._table.setItem(row, col, item)

    def _rebuild_table(self, result: SearchResult) -> None:
        """Redraw the table best-first, failures last."""
        self._table.setRowCount(0)
        for rank, trial in enumerate(result.ranked(), start=1):
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._set_row(row, str(rank), f"{float(trial.score):.4f}",
                          self._fold_sd(trial), format_params(trial.params),
                          "ok", trial.params, None)
        for trial in result.failed:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._set_row(row, "-", "-", "-", format_params(trial.params),
                          "failed", trial.params, trial.error)
        if self._table.rowCount():
            self._table.selectRow(0)

    def selected_params(self) -> Optional[Dict[str, Any]]:
        """The configuration on the selected row, or the best one.

        :returns: the parameter dict, or None when nothing is selectable.
        """
        rows = self._table.selectionModel().selectedRows() \
            if self._table.selectionModel() else []
        if rows:
            item = self._table.item(rows[0].row(), 0)
            if item is not None:
                data = item.data(Qt.UserRole)
                if isinstance(data, dict):
                    return dict(data)
        if self._result is not None and self._result.best is not None:
            return dict(self._result.best.params)
        return None

    def _on_selection_changed(self) -> None:
        """Enable Apply only while a real configuration is selected."""
        self._apply_btn.setEnabled(self.selected_params() is not None)

    # -- apply -------------------------------------------------------------

    def apply_selected(self) -> bool:
        """Push the selected configuration into the host's settings panel.

        :returns: True when a configuration was handed to the callback.
        """
        params = self.selected_params()
        if params is None:
            self._status.setText(
                "Select a row first — there is no configuration to apply.")
            return False
        if self._apply_cb is None:
            self._status.setText(
                "This panel is not attached to a settings panel, so there is "
                "nowhere to apply the configuration.")
            return False
        try:
            self._apply_cb(dict(params))
        except Exception as exc:
            self._status.setText(f"Could not apply the configuration: {exc}")
            return False
        msg = f"Applied {format_params(params)} to the settings panel."
        if self._result is not None and self._result.partial:
            msg += (" Note: this came from a partial sweep — configurations "
                    "that were never evaluated may be better.")
        self._status.setText(msg)
        return True

    # -- preview -----------------------------------------------------------

    def _draw_preview(self, result: SearchResult) -> None:
        """Render the small-multiples panel for a finished sweep."""
        try:
            fig = build_panel_figure(result)
        except Exception as exc:
            LOG.debug("panel figure failed", exc_info=True)
            self._preview.setText(f"Could not draw the preview: {exc}")
            return
        if fig is None:
            self._preview.setPixmap(QPixmap())
            self._preview.setText("No trial produced a score to plot.")
            return
        pm = figure_to_pixmap(fig)
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass
        if pm.isNull():
            self._preview.setText("Could not render the preview.")
            return
        self._preview.setPixmap(pm)

    # -- lifecycle ---------------------------------------------------------

    @property
    def result(self) -> Optional[SearchResult]:
        """The most recent :class:`SearchResult`, if any."""
        return self._result

    def closeEvent(self, event):     # noqa: N802 (Qt naming)
        """Stop a running sweep before the widget is torn down.

        Destroying a QWidget whose QThread is still running aborts the process,
        which is exactly how the headless test suite would die.
        """
        worker = self._worker
        if worker is not None:
            try:
                worker.request_stop()
                worker.quit()
                worker.wait(3000)
            except Exception:
                LOG.debug("worker shutdown failed", exc_info=True)
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Host integration
# ---------------------------------------------------------------------------

def build_hyperparam_card(host):
    """Build the ``Hyperparameter search`` card + panel pair.

    Mirrors ``spacr.qt.screens.app_screen._build_live_preview_card``: it returns
    the pair without adding it to any layout, so the host screen can put it in
    whatever splitter it likes and start it hidden behind the toggle.

    :param host: the :class:`AppScreen` asking for the card; its ``app_key``
        selects the parameter set.
    :returns: ``(panel, card)``.
    """
    from ..widgets.card import Card
    card = Card(title="Hyperparameter search")
    panel = HyperparamPanel(getattr(host, "app_key", "umap"), card)
    card.body_layout.addWidget(panel)
    card.setMinimumHeight(320)
    return panel, card


def searchable(app_key: str) -> bool:
    """Whether a hyperparameter search exists for ``app_key``."""
    return app_key in APP_PARAMS
