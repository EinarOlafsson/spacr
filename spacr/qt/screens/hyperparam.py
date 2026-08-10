"""Hyperparameter search panel — the Live-Preview-shaped window for sweeps.

Structurally this is the sibling of :mod:`spacr.qt.widgets.live_preview`: a
self-contained :class:`QWidget` that owns its own controls, runs the expensive
work on a :class:`QThread`, streams results back over signals, reports every
failure inline (never in a modal — a QMessageBox hangs a headless run), and
pushes a chosen configuration back into the host screen's settings panel through
a callback the host registers. Host screens embed it with
:func:`build_hyperparam_card`, exactly the way the Mask screen embeds the live
preview, and toggle it with a label reading "Hyperparameter search" where the
Mask screen's says "Live".

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
import tempfile
import threading
from pathlib import Path
from dataclasses import dataclass, field
from io import BytesIO
from typing import (
    Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple,
)

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QIcon, QPalette, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
    QDoubleSpinBox, QGridLayout, QHBoxLayout, QHeaderView, QGroupBox, QLabel,
    QLineEdit, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter,
    QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from ..widgets.figure_grid import SearchFigureGrid
from ..widgets.toggle import Toggle

from ...hyperparam import (
    APP_CRITERIA, DEFAULT_SPACES, DEFAULT_UMAP_OBJECTIVE_WEIGHTS,
    LOWER_IS_BETTER, SearchResult, SearchSpace, Trial, UMAP_CRITERIA,
    MAX_WALK_CANDIDATES_PER_ROUND, UMAP_METRICS as _UMAP_METRICS,
    UMAP_WALK_PARAMETERS, umap_walk_axes, walk_neighbourhood,
    run_search_for_app, umap_metrics as _umap_metrics,
)
from ..theme import active_palette, css_color, make_transparent

LOG = logging.getLogger("spacr.qt.hyperparam")

#: Label the host screens put on the toggle where the Mask screen says "Live".
TOGGLE_TEXT = "Hyperparameter search"

#: Tooltip for that toggle.
TOGGLE_TOOLTIP = (
    "Click to open the hyperparameter search. It sweeps the parameters you "
    "list, scores every configuration with a criterion you name, and reports "
    "the spread as well as the winner — so you can see whether the winner is "
    "real or noise. Nothing is applied to your settings until you press Apply."
)

#: Re-exported from :mod:`spacr.hyperparam`, which is where the searchable
#: UMAP parameters and their ranges live. They were defined here while the
#: search was a two-axis grid this panel owned; the Walk engine needs them
#: too and must not import Qt.
UMAP_METRICS = _UMAP_METRICS
umap_metrics = _umap_metrics



#: Where a Walk starts on a parameter the panel has no field for. UMAP's
#: own defaults, so an axis added without a starting value begins where an
#: unconfigured run would have.
_UMAP_DEFAULT_STARTS: Dict[str, Any] = {
    "n_neighbors": 15, "min_dist": 0.1, "n_components": 2,
    "metric": "euclidean", "spread": 1.0, "set_op_mix_ratio": 1.0,
    "local_connectivity": 1, "repulsion_strength": 1.0,
    "negative_sample_rate": 5, "init": "spectral",
}

#: How to parse a Walk starting value, keyed the same way as APP_PARAMS.
_WALK_AXIS_KIND: Dict[str, str] = {
    "n_neighbors": "int", "min_dist": "float", "n_components": "int",
    "metric": "metric", "spread": "float", "set_op_mix_ratio": "float",
    "local_connectivity": "int", "repulsion_strength": "float",
    "negative_sample_rate": "int", "init": "str",
}


#: Apps this panel can search, with the parameters it offers and their types.
#: ``(setting_key, label, kind)`` — kind drives the inline validation, so a
#: typo lands as a sentence under the controls instead of a traceback.
APP_PARAMS: Dict[str, Tuple[Tuple[str, str, str], ...]] = {
    "umap": (
        ("n_neighbors", "n_neighbors", "int"),
        ("min_dist", "min_dist", "float"),
        ("metric", "metric", "metric"),
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
    # Activation sweeps the settings that change the MAP, not the model: which
    # method, which layer it hooks, how much the input is smoothed, and the
    # window / step counts the perturbation and path-integral methods take.
    "activation": (
        ("cam_type", "method", "str"),
        ("target_layer", "target_layer", "str"),
        ("smoothgrad_samples", "smoothgrad n", "int"),
        ("smoothgrad_sigma", "smoothgrad sigma", "float"),
        ("occlusion_window", "occlusion window", "int"),
        ("occlusion_stride", "occlusion stride", "int"),
        ("ig_steps", "IG steps", "int"),
        ("ig_baseline", "IG baseline", "str"),
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
        elif kind == "metric":
            allowed = umap_metrics()
            if p not in allowed:
                raise ValueError(
                    f"Parameter '{name}' takes a umap-learn metric; {p!r} "
                    f"is not one. Try: "
                    f"{', '.join(allowed[:6])}…"
                )
            out.append(p)
        else:
            out.append(p)
    return out


def format_params(params: Dict[str, Any]) -> str:
    """Render a configuration as ``k=v, k=v`` in sorted-key order."""
    return ", ".join(f"{k}={params[k]}" for k in sorted(params))


def format_scores(trial: Trial, keys: Sequence[str] = ()) -> str:
    """Render every criterion a trial recorded, one per line.

    An Activation sweep computes four criteria for every trial precisely
    because they disagree, so the row that shows only the ranked one is hiding
    the finding. The panel puts this on the row's tooltip and in the figure
    titles.

    :param trial: the trial to describe.
    :param keys: criteria to show first, in order; anything else the trial
        recorded that is a plain number follows.
    :returns: the multi-line text (empty when the trial recorded nothing).
    """
    extra = trial.extra_metrics or {}
    lines: List[str] = []
    seen = set()
    for key in keys:
        if key in extra and isinstance(extra[key], (int, float)):
            lines.append(f"{key} = {float(extra[key]):.4f}")
            seen.add(key)
    for key in sorted(extra):
        if key in seen or not isinstance(extra[key], (int, float)) \
                or isinstance(extra[key], bool):
            continue
        lines.append(f"{key} = {float(extra[key]):.4g}")
    verdict = extra.get("sanity_verdict")
    if isinstance(verdict, str) and verdict:
        lines.append(verdict)
    for key in (
        "cluster_structure_method", "cluster_counts", "objective_weights",
    ):
        value = extra.get(key)
        if value not in (None, "", [], {}):
            lines.append(f"{key} = {value}")
    return "\n".join(lines)


def criteria_disagree(result: SearchResult,
                      criteria: Sequence[str]) -> Optional[str]:
    """Say plainly when re-ranking by another criterion picks another winner.

    Ranking attribution methods has no ground truth, so the useful output is
    not the top row but whether the top row survives a change of criterion.
    When it does not, that is the result and it goes on the status line.

    :param result: the finished sweep.
    :param criteria: the criteria to re-rank by.
    :returns: the sentence, or None when every criterion agrees (or there is
        not enough recorded to tell).
    """
    ranked = result.ranked()
    if len(ranked) < 2:
        return None
    winners: Dict[str, str] = {}
    for name in criteria:
        scored = [t for t in result.successful
                  if isinstance(t.extra_metrics.get(name), (int, float))
                  and not isinstance(t.extra_metrics.get(name), bool)]
        if len(scored) < 2:
            continue
        reverse = name not in LOWER_IS_BETTER
        best = sorted(scored,
                      key=lambda t: (float(t.extra_metrics[name]), t.index),
                      reverse=reverse)[0]
        winners[name] = format_params(best.params)
    if len(set(winners.values())) <= 1:
        return None
    listed = "; ".join(f"{k} -> {v}" for k, v in winners.items())
    return (f"THE CRITERIA DISAGREE: {listed}. There is no ground truth for "
            f"attribution, so this is not a tie to be broken — it means the "
            f"configurations differ in which property they satisfy. Look at "
            f"the maps.")


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


def _apply_figure_theme(fig, palette: Dict[str, str]) -> None:
    """Match a Matplotlib result panel to its Qt container palette."""
    background = palette["surface_alt"]
    foreground = palette["fg"]
    muted = palette.get("fg_muted", foreground)
    border = palette.get("border", muted)
    fig.patch.set_facecolor(background)
    for text in fig.texts:
        text.set_color(foreground)
    for ax in fig.axes:
        ax.set_facecolor(background)
        ax.title.set_color(foreground)
        ax.xaxis.label.set_color(foreground)
        ax.yaxis.label.set_color(foreground)
        ax.tick_params(axis="both", colors=muted)
        for spine in ax.spines.values():
            spine.set_color(border)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(background)
            legend.get_frame().set_edgecolor(border)
            for text in legend.get_texts():
                text.set_color(foreground)


def build_panel_figure(
    result: SearchResult,
    max_panels: int = MAX_PANELS,
    palette: Optional[Dict[str, str]] = None,
):
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

    palette = dict(palette or active_palette())
    ranked = result.ranked()
    if not ranked:
        return None

    # Attribution sweeps first: the maps ARE the deliverable, and the four
    # scores go in every title so the panel shows the criteria disagreeing
    # rather than hiding it behind one ranking.
    attributed = [t for t in ranked
                  if t.extra_metrics.get("attribution") is not None]
    if attributed:
        shown = attributed[:max_panels]
        cols = min(4, len(shown))
        rows = (len(shown) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.2 * rows),
                                 squeeze=False)
        for ax in axes.ravel():
            ax.set_axis_off()
        for i, trial in enumerate(shown):
            ax = axes[i // cols][i % cols]
            ax.set_axis_on()
            att = trial.extra_metrics["attribution"]
            heat = getattr(att, "map", att)
            ax.imshow(heat, cmap="jet")
            ax.set_xticks([])
            ax.set_yticks([])
            extra = trial.extra_metrics
            bits = [f"{k}={float(extra[k]):.3f}"
                    for k in ("deletion_auc", "insertion_auc", "pointing_game",
                              "sanity_gap")
                    if isinstance(extra.get(k), (int, float))
                    and not isinstance(extra.get(k), bool)]
            ax.set_title(f"{format_params(trial.params)}\n" + "  ".join(bits),
                         fontsize=6)
        fig.suptitle(
            f"{len(shown)} of {len(ranked)} attribution maps, ranked by "
            f"{result.metric} — deletion wants a LOW number, insertion and "
            f"pointing a high one; they disagree on purpose",
            fontsize=8)
        fig.tight_layout()
        _apply_figure_theme(fig, palette)
        return fig

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
            point_count = len(emb)
            ax.scatter([p[0] for p in emb], [p[1] for p in emb],
                       c=list(range(point_count)), cmap="viridis",
                       s=4, alpha=0.8, edgecolors="none")
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
        _apply_figure_theme(fig, palette)
        return fig

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    xs = list(range(1, len(ranked) + 1))
    ys = [float(t.score) for t in ranked]
    ax.plot(xs, ys, "o-", markersize=4, color=palette["accent"])
    noise, source = result.noise_level()
    if noise:
        best = ys[0]
        lo = best - noise if result.higher_is_better else best
        hi = best if result.higher_is_better else best + noise
        ax.axhspan(min(lo, hi), max(lo, hi), alpha=0.18,
                   color=palette["accent"],
                   label=f"within noise ({source})")
        ax.legend(fontsize=7)
    ax.set_xlabel("rank")
    ax.set_ylabel(result.metric)
    ax.set_title(f"{len(ranked)} configurations by {result.metric}",
                 fontsize=9)
    fig.tight_layout()
    _apply_figure_theme(fig, palette)
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
    adaptive: bool = False
    #: Which parameters the Walk searches. Empty means the original two,
    #: so an existing request is unchanged by the space becoming choosable.
    walk_parameters: Tuple[str, ...] = ()
    walk_resolutions: Dict[str, int] = field(default_factory=dict)
    n_neighbors_step: int = 1
    min_dist_step: float = 0.05
    min_improvement: float = 0.0
    stability_repeats: int = 3
    objective_weights: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_UMAP_OBJECTIVE_WEIGHTS)
    )
    seed: int = 0
    n_folds: int = 5
    resume: bool = False



def render_trial_figure(trial: Trial, metric: str, png_path: str) -> bool:
    """Draw ONE trial's embedding and write it to ``png_path``.

    Pure matplotlib, no Qt, so this is safe to call from a worker thread --
    which is the whole point. Fifty embeddings drawn on the GUI thread is a
    fifty-second freeze, and the search is the one place where the user most
    wants to keep interacting (to stop it).

    Returns False when the trial has no embedding to draw: a classifier
    sweep has none, and a failed UMAP trial has none either. The caller
    treats that as "no figure for this cell" rather than as an error.
    """
    embedding = trial.extra_metrics.get("embedding")
    if embedding is None or trial.score is None:
        return False
    import matplotlib
    matplotlib.use("Agg", force=False)
    from matplotlib.figure import Figure

    figure = Figure(figsize=(3.2, 2.6))
    axis = figure.add_subplot(111)
    count = len(embedding)
    axis.scatter([point[0] for point in embedding],
                 [point[1] for point in embedding],
                 c=list(range(count)), cmap="viridis",
                 s=4, alpha=0.8, edgecolors="none")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(f"{format_params(trial.params)}\n"
                   f"{metric}={float(trial.score):.3f}", fontsize=7)
    figure.tight_layout()
    from ..widgets.figure_queue import render_figure_to_png
    return bool(render_figure_to_png(figure, png_path))


class _SearchWorker(QThread):
    """Runs one sweep in the background, streaming trials as they complete."""

    # (Trial, completed, total, png_path). The path is rendered HERE, on
    # the worker thread, and is "" when the trial has no embedding.
    trial_ready = Signal(object, int, int, str)
    search_done = Signal(object, str)        # (SearchResult or None, error)

    def __init__(self, request: SearchRequest, search_fn=None, parent=None):
        """Store the request and the (optionally injected) search function."""
        super().__init__(parent)
        self._request = request
        self._search_fn = search_fn
        self._stop = threading.Event()
        self._metric = str(getattr(request, "criterion", "score") or "score")
        try:
            self._figure_dir = Path(tempfile.mkdtemp(prefix="spacr-search-"))
        except Exception:                       # pragma: no cover - defensive
            self._figure_dir = None
        # QThread.finished is the lifecycle boundary the panel must wait for.
        # The result signal is emitted just before QThread.run() returns, so it
        # is too early to drop the final Python reference to this object.
        self.result: Optional[SearchResult] = None
        self.error = ""
        self.completion_ready = False


    def _emit_trial(self, trial, done: int, total: int) -> None:
        """Render this trial, then announce it. Runs on the worker thread.

        A bound method rather than a lambda: the rendering has to happen
        before the signal is emitted, and doing it here keeps every
        matplotlib call off the GUI thread.
        """
        path = ""
        try:
            if self._figure_dir is not None:
                target = self._figure_dir / f"trial_{trial.index:04d}.png"
                if render_trial_figure(trial, self._metric, str(target)):
                    path = str(target)
        except Exception:
            # A figure is decoration; a search that dies because a plot
            # failed would lose hours of real work. INVARIANTS 10.
            LOG.debug("could not render trial %s", trial.index, exc_info=True)
        self.trial_ready.emit(trial, done, total, path)

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
            self.result = fn(req, self._emit_trial, self._stop.is_set)
        except Exception as exc:
            LOG.info("hyperparameter search failed: %s", exc, exc_info=True)
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.completion_ready = True
            # Kept for callers that use the private worker directly. The panel
            # deliberately consumes the stored payload from ``finished``.
            self.search_done.emit(self.result, self.error)


def _default_search_fn(request: SearchRequest, on_trial, should_stop) -> SearchResult:
    """Dispatch a request to :func:`spacr.hyperparam.run_search_for_app`."""
    return run_search_for_app(
        request.app_key, request.settings, request.space,
        criterion=request.criterion, mode=request.mode,
        n_trials=request.n_trials, adaptive=request.adaptive,
        walk_parameters=(list(request.walk_parameters)
                         if request.walk_parameters else None),
        walk_resolutions=dict(request.walk_resolutions or {}),
        n_neighbors_step=request.n_neighbors_step,
        min_dist_step=request.min_dist_step,
        min_improvement=request.min_improvement,
        stability_repeats=request.stability_repeats,
        objective_weights=request.objective_weights,
        seed=request.seed, n_folds=request.n_folds,
        on_trial=on_trial, should_stop=should_stop,
        resume=request.resume)


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


def _parse_walk_start(name: str, text: Any) -> Any:
    """One Walk starting value, typed the way its axis expects."""
    values = parse_values(str(text), _WALK_AXIS_KIND.get(name, "float"), name)
    if len(values) != 1:
        raise ValueError(
            f"The Walk starts from one value of {name}, not {len(values)}.")
    return values[0]


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
        self._settings_provider: Optional[
            Callable[[], Dict[str, Any]]
        ] = None
        self._value_edits: Dict[str, QLineEdit] = {}
        self._adaptive_grid_text: Dict[str, str] = {}
        self._walk_axes: Dict[str, Dict[str, Any]] = {}
        self._settings_dialog: Optional["UmapSearchSettingsDialog"] = None
        self._build_ui()

    # -- construction ------------------------------------------------------

    def _build_ui(self) -> None:
        """Lay out controls on the left, table + preview on the right."""
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        self._settings_panel = QGroupBox("Search & Plot Settings")
        self._settings_panel.setCheckable(True)
        self._settings_panel.setChecked(True)
        self._settings_panel.setToolTip(
            "Search-space, validation, reproducibility and result-plot "
            "controls. Collapse this drawer to give the results more room.")
        settings_layout = QVBoxLayout(self._settings_panel)
        settings_layout.setContentsMargins(8, 8, 8, 8)
        settings_layout.setSpacing(6)
        root.addWidget(self._settings_panel)

        # -- search space controls
        controls = QWidget()
        controls.setObjectName(
            "UmapHyperparamControls"
            if self.app_key == "umap" else "HyperparamControls"
        )
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
        settings_layout.addWidget(controls)

        # -- run controls. A grid keeps the settings dialog usable at normal
        # laptop widths; the old single horizontal row forced the popup wider
        # than the screen and made the first tab/title appear to overlap.
        run_grid = QGridLayout()
        run_grid.setContentsMargins(0, 0, 0, 0)
        run_grid.setHorizontalSpacing(6)
        run_grid.setVerticalSpacing(6)

        run_grid.addWidget(QLabel("criterion"), 0, 0)
        self._criterion = QComboBox()
        self._criterion.addItems(APP_CRITERIA[self.app_key])
        self._criterion.setToolTip(
            "The score used to rank trials. Hover here and read the explanation "
            "below before choosing: each criterion rewards a different kind "
            "of structure.")
        self._criterion.currentTextChanged.connect(
            self._update_criterion_explanation)
        run_grid.addWidget(self._criterion, 0, 1)

        run_grid.addWidget(QLabel("mode"), 0, 2)
        self._mode = QComboBox()
        self._mode.addItems(["grid", "random"])
        self._mode.setToolTip(
            "grid evaluates every combination; random samples n trials from "
            "the space with the seed below, which is reproducible.")
        run_grid.addWidget(self._mode, 0, 3)

        # "Walk", not "Adaptive 2x2". One name for one idea, shared with
        # the Gate Editor's clustering search -- a directional search
        # through hyperparameter space is the same thing in both modules
        # and was called two things. "2x2" was also a description of
        # today's two-parameter special case rather than of the design.
        self._adaptive = Toggle("Walk")
        self._adaptive.setVisible(self.app_key == "umap")
        self._adaptive.setToolTip(
            "Walk through hyperparameter space instead of sweeping a grid: "
            "each round steps in the direction that improved the score. The "
            "parameter fields above become single starting values. API: "
            "spacr.hyperparam.umap_search(adaptive=True).")
        self._adaptive.toggled.connect(self._on_adaptive_toggled)
        run_grid.addWidget(self._adaptive, 0, 4)
        self._walk_axes_button = QPushButton("Axes\u2026")
        self._walk_axes_button.setObjectName("WalkAxesButton")
        self._walk_axes_button.setVisible(self.app_key == "umap")
        self._walk_axes_button.setToolTip(
            "Choose which UMAP parameters the Walk searches and how finely. "
            "The default is n_neighbors and min_dist; every parameter that "
            "changes the structure of the embedding can be an axis.")
        self._walk_axes_button.clicked.connect(self.open_walk_axes)
        run_grid.addWidget(self._walk_axes_button, 0, 5)

        run_grid.addWidget(QLabel("n trials"), 1, 0)
        self._n_trials = QSpinBox()
        self._n_trials.setRange(1, 10_000)
        self._n_trials.setValue(12)
        self._n_trials.setToolTip(
            "How many configurations random mode draws. Adaptive mode uses "
            "the separate maximum-rounds field below. API: n_trials.")
        run_grid.addWidget(self._n_trials, 1, 1)

        self._n_folds_label = QLabel("folds")
        run_grid.addWidget(self._n_folds_label, 1, 2)
        self._n_folds = QSpinBox()
        self._n_folds.setRange(2, 50)
        self._n_folds.setValue(5)
        self._n_folds.setToolTip(
            "Cross-validation folds per trial. Folds are grouped by well, so "
            "crops from one well never straddle a split.")
        run_grid.addWidget(self._n_folds, 1, 3)
        if self.app_key in ("umap", "activation"):
            # Neither app cross-validates: UMAP fits one embedding per trial and
            # Activation attributes an already-trained model, so a fold count
            # would be a control that does nothing.
            self._n_folds_label.setVisible(False)
            self._n_folds.setVisible(False)

        run_grid.addWidget(QLabel("seed"), 1, 4)
        self._seed = QSpinBox()
        self._seed.setRange(0, 1_000_000)
        self._seed.setValue(0)
        self._seed.setToolTip(
            "Fixes the sampling, the folds and the reducer, so the same seed "
            "reproduces the same sweep.")
        run_grid.addWidget(self._seed, 1, 5)

        self._resume = Toggle("Resume checkpoint")
        self._resume.setVisible(self.app_key == "umap")
        self._resume.setToolTip(
            "Continue the compatible search checkpoint stored in the current "
            "project. Completed trials and embeddings are loaded; an "
            "interrupted adaptive round evaluates only its missing corners. "
            "Input data and material search settings must match. API: "
            "spacr.hyperparam.umap_search(resume=True).")
        run_grid.addWidget(self._resume, 2, 0, 1, 3)

        self._run_btn = QPushButton("Run search")
        self._run_btn.clicked.connect(self.run_search)
        run_grid.addWidget(self._run_btn, 3, 0, 1, 2)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.setToolTip(
            "Stop after the trial in flight. The trials already finished are "
            "kept and the result is marked partial.")
        self._stop_btn.clicked.connect(self.stop_search)
        run_grid.addWidget(self._stop_btn, 3, 2)

        self._apply_btn = QPushButton("Propagate settings")
        self._apply_btn.setEnabled(False)
        self._apply_btn.setToolTip(
            "Write the selected row's parameters into the settings panel. "
            "Nothing changes until you press this.")
        self._apply_btn.clicked.connect(self.apply_selected)
        run_grid.addWidget(self._apply_btn, 3, 3, 1, 3)
        run_grid.setColumnStretch(1, 2)
        run_grid.setColumnStretch(3, 2)
        run_grid.setColumnStretch(5, 2)
        settings_layout.addLayout(run_grid)

        self._criterion_help = QLabel()
        self._criterion_help.setWordWrap(True)
        self._criterion_help.setObjectName("HyperparamCriterionHelp")
        self._criterion_help.setToolTip(
            "A visible explanation of the selected ranking criterion. UMAP "
            "has no single score for whether a picture contains meaningful "
            "biological structure.")
        settings_layout.addWidget(self._criterion_help)

        # Multi-objective UMAP controls. These stay visible in the Search tab
        # so the composite is never a hidden formula, but are editable only
        # while that criterion is selected.
        self._multi_objective_controls = QWidget(self)
        self._multi_objective_controls.setObjectName(
            "UmapHyperparamControls"
            if self.app_key == "umap" else "HyperparamControls")
        objective_grid = QGridLayout(self._multi_objective_controls)
        objective_grid.setContentsMargins(0, 0, 0, 0)
        objective_grid.setHorizontalSpacing(6)
        objective_grid.setVerticalSpacing(6)
        repeats_label = QLabel("stability repeats")
        self._stability_repeats = QSpinBox()
        self._stability_repeats.setRange(2, 20)
        self._stability_repeats.setValue(3)
        repeats_tip = (
            "Independent, reproducibly seeded UMAP fits per configuration. "
            "Neighbour overlap across repeats is the stability objective. "
            "Runtime scales linearly; 3 is a practical default. API: "
            "spacr.hyperparam.embedding_stability."
        )
        repeats_label.setToolTip(repeats_tip)
        self._stability_repeats.setToolTip(repeats_tip)
        self._stability_repeats._spacr_setting_label = repeats_label
        objective_grid.addWidget(repeats_label, 0, 0)
        objective_grid.addWidget(self._stability_repeats, 0, 1)

        objective_specs = (
            (
                "neighborhood weight",
                "_neighborhood_weight",
                "umap_neighborhood_weight",
                DEFAULT_UMAP_OBJECTIVE_WEIGHTS[
                    "neighborhood_preservation"],
                "Weight for the geometric mean of trustworthiness and "
                "continuity. Both invented and lost neighbours are penalized.",
            ),
            (
                "stability weight",
                "_stability_weight",
                "umap_stability_weight",
                DEFAULT_UMAP_OBJECTIVE_WEIGHTS["stability"],
                "Weight for repeat-to-repeat nearest-neighbour agreement.",
            ),
            (
                "cluster weight",
                "_cluster_weight",
                "umap_cluster_structure_weight",
                DEFAULT_UMAP_OBJECTIVE_WEIGHTS["cluster_structure"],
                "Weight for positive silhouette structure. Supplied labels "
                "are used when available; otherwise spaCR discovers a "
                "reproducible 2–8 cluster partition.",
            ),
        )
        for offset, (
            label_text, attr, setting_key, default, detail,
        ) in enumerate(objective_specs, start=1):
            label = QLabel(label_text)
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 100.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.05)
            spin.setValue(default)
            tip = (
                f"{detail} Weights are normalized to sum to one; at least one "
                "must be positive. API: "
                "spacr.hyperparam.umap_objective_scores."
            )
            label.setToolTip(tip)
            spin.setToolTip(tip)
            spin._spacr_setting_label = label
            spin.setProperty("settingKey", setting_key)
            setattr(self, attr, spin)
            row, column = divmod(offset, 2)
            objective_grid.addWidget(label, row, column * 2)
            objective_grid.addWidget(spin, row, column * 2 + 1)
        objective_grid.setColumnStretch(1, 1)
        objective_grid.setColumnStretch(3, 1)
        self._multi_objective_controls.setVisible(self.app_key == "umap")
        settings_layout.addWidget(self._multi_objective_controls)
        self._update_criterion_explanation(self._criterion.currentText())

        # -- adaptive UMAP controls. Blank means the documented API default.
        adaptive_row = QHBoxLayout()
        adaptive_row.setSpacing(6)
        self._adaptive_controls = QWidget(self)
        self._adaptive_controls.setObjectName(
            "UmapHyperparamControls"
            if self.app_key == "umap" else "HyperparamControls")
        adaptive_controls_layout = QGridLayout(self._adaptive_controls)
        adaptive_controls_layout.setContentsMargins(0, 0, 0, 0)
        adaptive_controls_layout.setHorizontalSpacing(6)
        adaptive_controls_layout.setVerticalSpacing(6)
        for index, (label_text, attr, placeholder, tooltip) in enumerate((
            (
                "n increment", "_adaptive_n_step", "1",
                "Integer distance tested on either side of n_neighbors. "
                "Blank uses 1. API: n_neighbors_step.",
            ),
            (
                "min_dist increment", "_adaptive_d_step", "0.05",
                "Distance tested on either side of min_dist. Blank uses 0.05. "
                "API: min_dist_step.",
            ),
            (
                "maximum rounds", "_adaptive_rounds", "100",
                "Maximum complete 2×2 rounds. Blank uses 100. Search stops "
                "earlier when a round does not improve the score. "
                "API: n_trials in adaptive mode.",
            ),
            (
                "minimum improvement", "_adaptive_improvement", "0",
                "Score gain required to continue. Blank uses 0, so any strict "
                "improvement continues and a tie/stall stops. "
                "API: min_improvement.",
            ),
        )):
            label = QLabel(label_text)
            edit = QLineEdit()
            edit.setPlaceholderText(placeholder)
            label.setToolTip(tooltip)
            edit.setToolTip(tooltip)
            setattr(self, attr, edit)
            grid_row, grid_column = divmod(index, 2)
            adaptive_controls_layout.addWidget(
                label, grid_row, grid_column * 2)
            adaptive_controls_layout.addWidget(
                edit, grid_row, grid_column * 2 + 1)
        adaptive_controls_layout.setColumnStretch(1, 1)
        adaptive_controls_layout.setColumnStretch(3, 1)
        adaptive_row.addWidget(self._adaptive_controls)
        settings_layout.addLayout(adaptive_row)
        self._adaptive_controls.setVisible(self.app_key == "umap")
        self._on_adaptive_toggled(False)

        plot_row = QHBoxLayout()
        plot_label = QLabel("maximum graph panels")
        self._max_panels = QSpinBox()
        self._max_panels.setRange(1, 48)
        self._max_panels.setValue(MAX_PANELS)
        self._max_panels.setToolTip(
            "Maximum successful trials drawn in the result graph. The table "
            "still contains every trial. API: build_panel_figure(max_panels=…).")
        plot_label.setToolTip(self._max_panels.toolTip())
        self._max_panels._spacr_setting_label = plot_label
        plot_row.addWidget(plot_label)
        plot_row.addWidget(self._max_panels)
        plot_row.addStretch(1)
        settings_layout.addLayout(plot_row)

        # Match Measure Live: keep the card focused on results and put the
        # complete control set behind a settings button in a separate window.
        root.removeWidget(self._settings_panel)
        self._settings_panel.hide()
        compact_actions = QHBoxLayout()
        self._compact_run_btn = QPushButton("Run search")
        self._compact_run_btn.clicked.connect(self.run_search)
        self._compact_stop_btn = QPushButton("Stop")
        self._compact_stop_btn.setObjectName("DangerButton")
        self._compact_stop_btn.setProperty("buttonActionRole", "negative")
        self._compact_stop_btn.setEnabled(False)
        self._compact_stop_btn.setToolTip(
            "Stop after the UMAP trial currently in flight. The completed "
            "trials are retained and marked as a partial search.")
        self._compact_stop_btn.clicked.connect(self.stop_search)
        self._settings_btn = QPushButton(
            "UMAP settings…" if self.app_key == "umap"
            else "Search settings…")
        self._settings_btn.setToolTip(
            "Open the tabbed search, graph and module settings window. "
            "This follows Measure Live's Crop settings pattern.")
        self._settings_btn.clicked.connect(self.open_settings)
        compact_actions.addWidget(self._compact_run_btn)
        compact_actions.addWidget(self._compact_stop_btn)
        compact_actions.addWidget(self._settings_btn)
        compact_actions.addStretch(1)
        root.insertLayout(0, compact_actions)

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

        # Two views of the same sweep, stacked: the GRID fills a cell at a
        # time while the search runs, and the summary figure replaces it at
        # the end. Live steering is the grid's job -- seeing the first few
        # embeddings is what tells you the range is wrong before the other
        # fifty run -- and the summary is the deliverable.
        self._preview_stack = QWidget(self)
        self._preview_stack.setObjectName("SearchPreviewStack")
        make_transparent(self._preview_stack)
        preview_column = QVBoxLayout(self._preview_stack)
        preview_column.setContentsMargins(0, 0, 0, 0)
        preview_column.setSpacing(0)

        self._figure_grid = SearchFigureGrid(parent=self._preview_stack)
        self._figure_grid.setVisible(False)
        self._figure_grid.cell_clicked.connect(self._open_trial_figure)
        preview_column.addWidget(self._figure_grid, 1)

        self._preview = QLabel("No search has been run yet.")
        self._preview.setAlignment(Qt.AlignCenter)
        self._preview.setMinimumWidth(220)
        self._preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._preview.setWordWrap(True)
        preview_column.addWidget(self._preview, 1)
        split.addWidget(self._preview_stack)
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

    def set_settings_provider(
        self,
        provider: Optional[Callable[[], Dict[str, Any]]],
    ) -> None:
        """Register a callback returning the host's current module settings.

        The main settings form remains editable while this panel or its popup
        is open. Reading it immediately before each search prevents a source
        path dropped after the panel opened from being lost in a stale
        snapshot.

        :param provider: zero-argument callback returning a settings mapping.
        """
        self._settings_provider = provider

    def set_search_fn(self, fn) -> None:
        """Override the search backend.

        :param fn: ``fn(request, on_trial, should_stop) -> SearchResult``.
        """
        self._search_fn = fn

    def open_settings(self) -> None:
        """Open or focus the tabbed search/module settings window."""
        dialog = self._settings_dialog
        if dialog is not None and dialog.isVisible():
            dialog.raise_()
            dialog.activateWindow()
            return
        dialog = UmapSearchSettingsDialog(self)
        self._settings_dialog = dialog
        dialog.finished.connect(self._on_settings_closed)
        dialog.show()

    def _on_settings_closed(self, *_args) -> None:
        self._settings_panel.setParent(self)
        self._settings_panel.hide()
        self._settings_dialog = None

    def _update_criterion_explanation(self, criterion: str) -> None:
        """Explain what the selected score calls 'structure'."""
        if self.app_key == "umap":
            detail = UMAP_CRITERIA.get(criterion, "")
            recommendation = {
                "multi_objective": (
                    "Recommended when you want unknown structure: inspect the "
                    "Pareto front rather than treating the composite top row "
                    "as a uniquely correct answer."),
                "trustworthiness": (
                    "Best default for finding local structure without "
                    "inventing apparent neighbours."),
                "continuity": (
                    "Use when preserving existing neighbourhoods matters most; "
                    "it may crowd unrelated points together."),
                "silhouette": (
                    "Use only to test separation of labels you already have; "
                    "it does not discover unknown structure."),
            }.get(criterion, "")
            self._criterion_help.setText(
                f"{criterion}: {detail} {recommendation}")
            self._criterion.setToolTip(self._criterion_help.text())
            controls = getattr(self, "_multi_objective_controls", None)
            if controls is not None:
                controls.setEnabled(criterion == "multi_objective")
            return
        self._criterion_help.setText(
            f"{criterion} ranks the trials. See the control tooltip and the "
            "per-trial score tooltips in the results table.")

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

    def _on_adaptive_toggled(self, checked: bool) -> None:
        """Switch the UMAP fields between grid lists and one local centre."""
        checked = bool(checked) and self.app_key == "umap"
        controls = getattr(self, "_adaptive_controls", None)
        if controls is not None:
            # Keep labels and their API dots active even when adaptive search
            # is off; only the value fields are unavailable.
            controls.setEnabled(True)
            for edit in (
                    self._adaptive_n_step, self._adaptive_d_step,
                    self._adaptive_rounds, self._adaptive_improvement):
                edit.setEnabled(checked)
        button = getattr(self, "_walk_axes_button", None)
        if button is not None:
            button.setEnabled(checked)
        if hasattr(self, "_mode"):
            self._mode.setEnabled(not checked)
        if hasattr(self, "_n_trials"):
            self._n_trials.setEnabled(not checked)
        if not checked or not self._value_edits:
            if not checked and self._adaptive_grid_text:
                for key, text in self._adaptive_grid_text.items():
                    if key in self._value_edits:
                        self._value_edits[key].setText(text)
            return
        for key in ("n_neighbors", "min_dist"):
            edit = self._value_edits.get(key)
            if edit is None:
                continue
            text = edit.text().strip()
            if "," in text:
                self._adaptive_grid_text[key] = text
                value = self._settings.get(key)
                if value is None:
                    defaults = DEFAULT_SPACES.get("umap", {}).get(key, ())
                    value = defaults[0] if defaults else ""
                edit.setText(str(value))

    def current_adaptive_space(self) -> SearchSpace:
        """Return one UMAP starting centre, using settings defaults if blank."""
        params: Dict[str, List[Any]] = {}
        defaults = {"n_neighbors": 1000, "min_dist": 0.1,
                    "metric": "euclidean"}
        kinds = {"n_neighbors": "int", "min_dist": "float", "metric": "str"}
        for key in ("n_neighbors", "min_dist", "metric"):
            edit = self._value_edits[key]
            text = edit.text().strip()
            if not text:
                value = self._settings.get(key, defaults[key])
            else:
                values = parse_values(text, kinds[key], key)
                if len(values) != 1:
                    fallback = self._settings.get(key, defaults[key])
                    raise ValueError(
                        "Adaptive 2×2 optimization needs one starting value "
                        f"for {key}; leave it blank to use {fallback!r}.")
                value = values[0]
            params[key] = [value]
        return SearchSpace(params)

    def adaptive_parameters(self) -> Tuple[int, int, float, float]:
        """Parse adaptive increments, rounds and convergence threshold."""
        def _number(edit, default, cast, label):
            text = edit.text().strip()
            if not text:
                return default
            try:
                value = cast(text)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{label} must be a number; leave it blank to use "
                    f"{default}.") from exc
            return value

        n_step = _number(self._adaptive_n_step, 1, int,
                         "n_neighbors increment")
        d_step = _number(self._adaptive_d_step, 0.05, float,
                         "min_dist increment")
        rounds = _number(self._adaptive_rounds, 100, int, "maximum rounds")
        improvement = _number(
            self._adaptive_improvement, 0.0, float, "minimum improvement")
        if n_step < 1 or d_step <= 0 or rounds < 1 or improvement < 0:
            raise ValueError(
                "Adaptive increments and maximum rounds must be positive; "
                "minimum improvement must be zero or greater.")
        return rounds, n_step, d_step, improvement

    # -- search space ------------------------------------------------------


    # -- Walk axes ---------------------------------------------------------

    def walk_axes(self) -> Dict[str, Dict[str, Any]]:
        """The chosen Walk axes, ``{name: {'start', 'resolution'}}``.

        Empty means the Walk uses the two parameters it always used, which
        is what an untouched panel should do.
        """
        return {name: dict(spec) for name, spec in self._walk_axes.items()}

    def set_walk_axes(self, axes: Mapping[str, Mapping[str, Any]]) -> None:
        """Replace the chosen Walk axes."""
        cleaned: Dict[str, Dict[str, Any]] = {}
        for name, spec in (axes or {}).items():
            if name not in UMAP_WALK_PARAMETERS:
                raise ValueError(
                    f"Not a searchable UMAP parameter: {name!r}. Searchable: "
                    f"{sorted(UMAP_WALK_PARAMETERS)}.")
            cleaned[name] = {
                "start": str(spec.get("start", "")).strip(),
                "resolution": max(2, int(spec.get("resolution", 2) or 2)),
            }
        self._walk_axes = cleaned

    def walk_start_for(self, name: str) -> str:
        """Where the Walk should start on ``name``, as text for a field.

        The main search field wins when it holds one value, then the run's
        settings, then UMAP's own default. A starting point the user can
        see somewhere else in the panel is the one they expect.
        """
        edit = self._value_edits.get(name)
        if edit is not None:
            text = edit.text().strip()
            if text and "," not in text:
                return text
        value = self._settings.get(name)
        if value is not None:
            return str(value)
        return str(_UMAP_DEFAULT_STARTS.get(name, ""))

    def open_walk_axes(self) -> Optional["WalkAxesDialog"]:
        """Open the axis picker. Returns the dialog, or None when not UMAP."""
        if self.app_key != "umap":
            return None
        dialog = WalkAxesDialog(self)
        if dialog.exec() == QDialog.Accepted:
            self.set_walk_axes(dialog.selection())
        return dialog

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
        walking = (self.app_key == "umap"
                   and getattr(self, "_adaptive", None) is not None
                   and self._adaptive.isChecked())
        if walking and self._walk_axes:
            # A Walk needs a starting POINT, so each chosen axis enters the
            # space as exactly one value. Parameters that are not axes keep
            # whatever the fields hold, which is how a fixed metric or a
            # fixed n_components travels with the walk.
            for name, spec in self._walk_axes.items():
                text = str(spec.get("start", "")).strip()
                if not text:
                    text = self.walk_start_for(name)
                if not text:
                    raise ValueError(
                        f"The Walk searches {name} but has no value to start "
                        "from. Give it one in the Axes dialog.")
                kind = _WALK_AXIS_KIND.get(name, "float")
                params[name] = parse_values(text, kind, name)
        if not params:
            raise ValueError(
                "Nothing to search: fill in at least one parameter with at "
                "least one value, e.g. n_neighbors = 5, 15, 50.")
        if walking:
            multiple = sorted(k for k, v in params.items() if len(v) > 1)
            if multiple:
                raise ValueError(
                    "A Walk starts from one point, so each parameter takes a "
                    f"single value. These hold more than one: "
                    f"{', '.join(multiple)}.")
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
        if self._settings_provider is not None:
            try:
                current = self._settings_provider()
                if not isinstance(current, dict):
                    raise TypeError(
                        "the module settings provider did not return a dict")
                self._settings = dict(current)
            except Exception as exc:
                self._status.setText(
                    f"Could not read the current module settings: {exc}")
                return False
        adaptive = self.app_key == "umap" and self._adaptive.isChecked()
        try:
            space = (
                self.current_adaptive_space()
                if adaptive else self.current_space())
            if adaptive:
                rounds, n_step, d_step, improvement = (
                    self.adaptive_parameters())
            else:
                rounds, n_step, d_step, improvement = (
                    int(self._n_trials.value()), 1, 0.05, 0.0)
        except ValueError as exc:
            self._status.setText(str(exc))
            return False
        objective_weights = dict(DEFAULT_UMAP_OBJECTIVE_WEIGHTS)
        stability_repeats = 3
        if self.app_key == "umap":
            objective_weights = {
                "neighborhood_preservation":
                    float(self._neighborhood_weight.value()),
                "stability": float(self._stability_weight.value()),
                "cluster_structure": float(self._cluster_weight.value()),
            }
            stability_repeats = int(self._stability_repeats.value())
            if (
                self._criterion.currentText() == "multi_objective"
                and sum(objective_weights.values()) <= 0
            ):
                self._status.setText(
                    "At least one multi-objective UMAP weight must be positive."
                )
                return False

        request = SearchRequest(
            app_key=self.app_key,
            space=space,
            settings=dict(self._settings),
            criterion=self._criterion.currentText(),
            mode=self._mode.currentText(),
            n_trials=rounds,
            adaptive=adaptive,
            walk_parameters=(tuple(self._walk_axes) if adaptive else ()),
            walk_resolutions={
                name: int(spec.get("resolution", 2))
                for name, spec in self._walk_axes.items()} if adaptive else {},
            n_neighbors_step=n_step,
            min_dist_step=d_step,
            min_improvement=improvement,
            stability_repeats=stability_repeats,
            objective_weights=objective_weights,
            seed=int(self._seed.value()),
            n_folds=int(self._n_folds.value()),
            resume=self.app_key == "umap" and self._resume.isChecked(),
        )
        self._result = None
        self._live_trials = []
        self._table.setRowCount(0)
        self._apply_btn.setEnabled(False)
        self._notes.setText("")
        self._preview.setText("Running…")
        self._preview.setPixmap(QPixmap())
        if self._figure_grid is not None:
            self._figure_grid.clear()
            # Axes from the SPACE, set before the first figure lands, so the
            # grid does not rearrange itself as results arrive.
            try:
                self._figure_grid.set_parameters(list(space.params))
            except Exception:
                self._figure_grid.set_parameters([])
            self._figure_grid.setVisible(True)
            self._preview.setVisible(False)
        search_label = (
            f"adaptive 2×2 search over at most {request.n_trials} rounds"
            if request.adaptive
            else f"{request.mode} search over {space.size()}")
        self._status.setText(
            f"Running {search_label}, ranked by "
            f"{request.criterion}…")

        worker = _SearchWorker(request, self._search_fn, self)
        worker.trial_ready.connect(self._on_trial_ready)
        # NOT worker.deleteLater. `finished` is emitted from inside the worker
        # thread, so scheduling the object's deletion there hands C++ a second
        # owner for an object Python already owns, and the two race — see the
        # measured account in spacr.qt.bridge.make_thread. The relay below is a
        # bound method, so the connection keeps `self` alive rather than a
        # lambda closure Qt cannot introspect, and the worker is freed when the
        # panel drops its reference on the GUI thread.
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        self._set_search_running(True)
        worker.start()
        return True

    def stop_search(self) -> None:
        """Ask the running sweep to stop; the result comes back partial."""
        worker = self._worker
        if worker is None or not worker.isRunning():
            self._status.setText("No search is running.")
            return
        worker.request_stop()
        # A stop request is asynchronous: keep the pressed negative button
        # solid red and prevent repeat requests until the worker exits.
        from ..button_roles import set_button_busy
        sender = self.sender()
        for button in (self._stop_btn, self._compact_stop_btn):
            button.setEnabled(False)
            set_button_busy(button, button is sender)
        self._status.setText(
            "Stopping after the trial in flight — the finished trials are "
            "kept and the result is marked partial.")

    # -- signal handlers ---------------------------------------------------

    def _set_search_running(self, running: bool) -> None:
        """Synchronize every Run/Stop control, including the popup footer."""
        from ..button_roles import set_button_busy
        run_buttons = [self._run_btn, self._compact_run_btn]
        dialog = self._settings_dialog
        if dialog is not None:
            footer_run = getattr(dialog, "_run_btn", None)
            if footer_run is not None:
                run_buttons.append(footer_run)
        for button in run_buttons:
            button.setEnabled(not running)
        for button in (self._stop_btn, self._compact_stop_btn):
            set_button_busy(button, False)
            button.setEnabled(running)

    def _on_worker_finished(self) -> None:
        """Consume a result only after the QThread has completely exited.

        A worker's result signal is emitted from inside ``run`` and therefore
        precedes ``QThread.finished``. Clearing ``self._worker`` in that earlier
        slot let a new search start while the old QThread was still unwinding;
        repeated UMAP searches could consequently stall or lose their worker.
        """
        sender = self.sender()
        worker = sender if isinstance(sender, _SearchWorker) else self._worker
        # A stale completion must never re-enable controls belonging to a newer
        # run. This is defensive now that starts are serialized at ``finished``.
        if worker is None or worker is not self._worker:
            return
        self._worker = None
        self._set_search_running(False)
        if not worker.completion_ready:
            self._on_search_done(
                None, "Search worker exited without returning a result.")
            return
        self._on_search_done(worker.result, worker.error)

    def _on_trial_ready(self, trial: Trial, done: int, total: int,
                        png_path: str = "") -> None:
        """Append one finished trial to the table as the sweep progresses.

        :param png_path: a figure the WORKER already rendered, or "". The
            panel only loads and places it -- rendering here would be on
            the GUI thread.
        """
        self._live_trials.append(trial)
        if png_path and self._figure_grid is not None:
            try:
                self._figure_grid.add_figure(png_path, dict(trial.params))
            except Exception:
                LOG.debug("could not place the trial figure", exc_info=True)
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._set_row(row, str(trial.index + 1),
                      "-" if trial.score is None else f"{trial.score:.4f}",
                      self._fold_sd(trial),
                      format_params(trial.params),
                      "failed" if trial.error else "ok",
                      trial.params, trial.error, trial)
        self._status.setText(
            f"{done} of {total} configurations evaluated"
            + (f" — last one failed: {trial.error}" if trial.error else ""))

    def _on_search_done(self, result: Optional[SearchResult], err: str) -> None:
        """Rebuild the table in ranked order and draw the preview."""
        if err:
            self._status.setText(f"Search failed: {err}")
            self._preview.setText("Search failed.")
            return
        if result is None:
            self._status.setText(
                "Search failed: the worker returned no result.")
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
        pareto = result.pareto_front()
        if pareto:
            summary.append(
                f"Pareto front: {len(pareto)} non-dominated "
                "configuration(s); inspect their objective tooltips before "
                "propagating one."
            )
        disagreement = criteria_disagree(result,
                                         APP_CRITERIA.get(self.app_key, ()))
        if disagreement:
            summary.append(disagreement)
        self._status.setText(" ".join(summary))
        self._show_summary_instead_of_grid()
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
                 error: Optional[str], trial: Optional[Trial] = None) -> None:
        """Write one table row and stash the config on the first cell.

        Every criterion the trial recorded goes on the row's tooltip, not just
        the one the table is ranked by: for an Activation sweep the other three
        are the reason the ranking should not be read as a verdict.
        """
        cells = (rank, score, sd, params, error or status)
        detail = format_scores(trial, APP_CRITERIA.get(self.app_key, ())) \
            if trial is not None else ""
        for col, text in enumerate(cells):
            item = QTableWidgetItem(text)
            if col == 0:
                item.setData(Qt.UserRole, dict(param_dict))
                if trial is not None:
                    item.setData(Qt.UserRole + 1, dict(trial.extra_metrics))
            tip = error or detail
            if tip:
                item.setToolTip(tip)
            self._table.setItem(row, col, item)

    def _rebuild_table(self, result: SearchResult) -> None:
        """Redraw the table best-first, failures last."""
        self._table.setRowCount(0)
        pareto_ids = {id(trial) for trial in result.pareto_front()}
        for rank, trial in enumerate(result.ranked(), start=1):
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._set_row(row, str(rank), f"{float(trial.score):.4f}",
                          self._fold_sd(trial), format_params(trial.params),
                          (
                              "Pareto" if id(trial) in pareto_ids
                              else "dominated"
                          ) if result.objectives else "ok",
                          trial.params, None, trial)
        for trial in result.failed:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._set_row(row, "-", "-", "-", format_params(trial.params),
                          "failed", trial.params, trial.error, trial)
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

    def _open_trial_figure(self, index: int) -> None:
        """Open the clicked trial's figure in the desktop viewer.

        Without this the per-trial figures are unreachable: they are
        written to a temporary directory and only ever seen as thumbnails,
        so a user who asked for PDFs got PDFs they could not open.
        `figure_path` hands back the vector PDF when one was written.
        """
        grid = getattr(self, "_figure_grid", None)
        if grid is None:
            return
        path = grid.figure_path(index)
        if not path:
            return
        try:
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        except Exception:
            LOG.debug("could not open %s", path, exc_info=True)

    def _show_summary_instead_of_grid(self) -> None:
        """Add the summary BESIDE the grid rather than in place of it.

        This used to hide the grid and show only the finished-sweep figure,
        which reproduced the exact complaint the grid was built to answer:
        "not all at the end in one large grid and as a large PNG". The
        per-trial figures appeared during the run and were then replaced by
        one big image the moment it finished, so the end state -- the only
        state a user who steps away ever sees -- was the old behaviour.

        Both are shown now. The grid stays, because it is the thing the
        user asked to look at and it is what makes a bad range obvious; the
        summary keeps its place below it, because it carries the ranking
        and the noise band that the individual panels cannot.
        """
        if self._figure_grid is not None and self._figure_grid.count():
            self._figure_grid.setVisible(True)
        self._preview.setVisible(True)

    def _draw_preview(self, result: SearchResult) -> None:
        """Render the small-multiples panel for a finished sweep."""
        try:
            fig = build_panel_figure(
                result, max_panels=int(self._max_panels.value()))
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
# Tabbed settings window — mirrors Measure Live's CropSettingsDialog
# ---------------------------------------------------------------------------

class UmapSearchSettingsDialog(QDialog):
    """Tabbed search and UMAP-graph settings for :class:`HyperparamPanel`."""

    def __init__(self, panel: HyperparamPanel):
        super().__init__(panel)
        self._panel = panel
        self._module_model = None
        self._module_keys = set()
        self.setObjectName("UmapSearchSettingsDialog")
        self.setWindowTitle(
            "UMAP settings" if panel.app_key == "umap"
            else "Hyperparameter search settings")
        outer = QVBoxLayout(self)
        self._tabs = QTabWidget(self)
        self._tabs.setObjectName("UmapSettingsTabs")
        outer.addWidget(self._tabs, 1)

        # A group box cannot safely be the tab page itself: its title notch and
        # frame share the tab pane's top-left origin. Put it on an ordinary
        # padded page instead, matching Measure Live's settings dialogs.
        self._search_page = QWidget(self)
        self._search_page.setObjectName("UmapSearchPage")
        search_layout = QVBoxLayout(self._search_page)
        search_layout.setContentsMargins(12, 12, 12, 12)
        panel._settings_panel.setParent(self._search_page)
        panel._settings_panel.setObjectName("UmapSearchGroup")
        panel._settings_panel.setTitle("Hyperparameter Search")
        panel._settings_panel.setCheckable(False)
        panel._settings_panel.show()
        search_layout.addWidget(panel._settings_panel)
        search_layout.addStretch(1)
        self._search_scroll = QScrollArea(self)
        self._search_scroll.setObjectName("UmapSearchScroll")
        self._search_scroll.setWidgetResizable(True)
        self._search_scroll.setFrameShape(QScrollArea.NoFrame)
        self._search_scroll.setWidget(self._search_page)
        self._tabs.addTab(self._search_scroll, "Search")

        if panel.app_key == "umap":
            self._build_umap_tabs()

        # The popup has one action row at its foot. The copies embedded in the
        # Search group were left over from when the panel itself was the whole
        # window and produced two Runs and two Propagates plus an unnecessary
        # middle Stop.
        panel._run_btn.hide()
        panel._stop_btn.hide()
        panel._apply_btn.hide()

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        close_button = buttons.button(QDialogButtonBox.Close)
        self._close_btn = close_button
        if close_button is not None:
            # Some platform themes put a red X on the standard Close button.
            # The semantic red outline/text already communicates the action.
            close_button.setIcon(QIcon())
        self._run_btn = QPushButton("Run search")
        self._run_btn.clicked.connect(panel.run_search)
        buttons.addButton(self._run_btn, QDialogButtonBox.ActionRole)
        self._propagate = QPushButton("Propagate settings", self)
        self._propagate.setToolTip(
            "Copy the current settings in this window into the main UMAP "
            "module settings once. A selected successful trial overrides the "
            "corresponding n_neighbors, min_dist and metric values.")
        self._propagate.clicked.connect(self.propagate_settings)
        buttons.addButton(self._propagate, QDialogButtonBox.ActionRole)
        buttons.rejected.connect(self.close)
        outer.addWidget(buttons)
        self._run_btn.setEnabled(
            panel._worker is None or not panel._worker.isRunning())

        palette = active_palette()
        bg = palette["bg"]
        field = palette["surface_alt"]
        fg = palette["fg"]
        border = palette["border"]
        # Only this popup: every settings surface is the theme's black canvas;
        # editable/value fields alone are lifted to dark gray. Do not alter the
        # application palette.
        self.setStyleSheet(
            f"""
            QDialog#UmapSearchSettingsDialog,
            QDialog#UmapSearchSettingsDialog QWidget,
            QDialog#UmapSearchSettingsDialog QWidget#UmapSearchPage,
            QDialog#UmapSearchSettingsDialog QWidget#UmapSettingsPage,
            QDialog#UmapSearchSettingsDialog QWidget#UmapHyperparamControls,
            QDialog#UmapSearchSettingsDialog QGroupBox,
            QDialog#UmapSearchSettingsDialog QScrollArea,
            QDialog#UmapSearchSettingsDialog QScrollArea::viewport {{
                background-color: {bg};
                color: {fg};
            }}
            QDialog#UmapSearchSettingsDialog QTabWidget#UmapSettingsTabs::pane {{
                background-color: {bg};
                border: 1px solid {border};
                top: -1px;
            }}
            QDialog#UmapSearchSettingsDialog QTabBar::tab {{
                background-color: {bg};
                color: {fg};
                border: 1px solid {border};
                padding: 7px 12px;
                margin-right: 2px;
            }}
            QDialog#UmapSearchSettingsDialog QTabBar::tab:selected,
            QDialog#UmapSearchSettingsDialog QTabBar::tab:hover {{
                background-color: {bg};
                color: {fg};
                border-color: {palette["accent"]};
            }}
            QDialog#UmapSearchSettingsDialog QGroupBox#UmapSearchGroup {{
                background-color: {bg};
                color: {fg};
                border: 1px solid {border};
                margin-top: 18px;
                padding-top: 10px;
            }}
            QDialog#UmapSearchSettingsDialog
            QGroupBox#UmapSearchGroup::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 2px 6px;
                background-color: {bg};
                color: {fg};
            }}
            QDialog#UmapSearchSettingsDialog
            QGroupBox#UmapSearchGroup QLabel,
            QDialog#UmapSearchSettingsDialog
            QWidget#UmapSettingsPage QLabel {{
                background-color: {bg};
                color: {fg};
            }}
            QDialog#UmapSearchSettingsDialog QLabel:disabled {{
                background-color: {bg};
                color: {fg};
            }}
            QDialog#UmapSearchSettingsDialog QLineEdit,
            QDialog#UmapSearchSettingsDialog QSpinBox,
            QDialog#UmapSearchSettingsDialog QDoubleSpinBox,
            QDialog#UmapSearchSettingsDialog QComboBox,
            QDialog#UmapSearchSettingsDialog QTableWidget,
            QDialog#UmapSearchSettingsDialog QAbstractItemView {{
                background-color: {field};
                color: {fg};
            }}
            QDialog#UmapSearchSettingsDialog QLineEdit:disabled,
            QDialog#UmapSearchSettingsDialog QSpinBox:disabled,
            QDialog#UmapSearchSettingsDialog QDoubleSpinBox:disabled,
            QDialog#UmapSearchSettingsDialog QComboBox:disabled {{
                background-color: {field};
                color: {fg};
            }}
            QDialog#UmapSearchSettingsDialog QLineEdit::placeholder {{
                color: {fg};
            }}
            QDialog#UmapSearchSettingsDialog QPushButton {{
                background-color: {bg};
                color: {fg};
            }}
            QDialog#UmapSearchSettingsDialog
            QPushButton[buttonActionRole="positive"] {{
                background-color: transparent;
                color: {palette["accent"]};
                border: 1px solid {palette["accent"]};
            }}
            QDialog#UmapSearchSettingsDialog
            QPushButton[buttonActionRole="positive"]:hover {{
                background-color: {css_color(palette["accent"], 0.18)};
            }}
            QDialog#UmapSearchSettingsDialog
            QPushButton[buttonActionRole="positive"]:pressed,
            QDialog#UmapSearchSettingsDialog
            QPushButton[buttonActionRole="positive"][buttonActionBusy="true"] {{
                background-color: {palette["accent"]};
                color: {bg};
            }}
            QDialog#UmapSearchSettingsDialog
            QPushButton[buttonActionRole="negative"] {{
                background-color: transparent;
                color: {palette["error"]};
                border: 1px solid {palette["error"]};
            }}
            QDialog#UmapSearchSettingsDialog
            QPushButton[buttonActionRole="negative"]:hover {{
                background-color: {css_color(palette["error"], 0.18)};
            }}
            QDialog#UmapSearchSettingsDialog
            QPushButton[buttonActionRole="negative"]:pressed,
            QDialog#UmapSearchSettingsDialog
            QPushButton[buttonActionRole="negative"][buttonActionBusy="true"] {{
                background-color: {palette["error"]};
                color: {bg};
            }}
            """
        )
        # Some platform styles apply disabled/placeholder opacity after QSS.
        # Pin every text palette role to white inside this dialog so labels and
        # field text remain white even when an adaptive control is inactive.
        white = QColor(fg)
        for widget in self.findChildren(QWidget):
            widget_palette = widget.palette()
            for group in (
                    QPalette.Active, QPalette.Inactive, QPalette.Disabled):
                for role in (
                        QPalette.WindowText, QPalette.Text,
                        QPalette.ButtonText, QPalette.PlaceholderText):
                    widget_palette.setColor(group, role, white)
            widget.setPalette(widget_palette)
        from .settings_model import install_api_tooltips
        search_tooltips = {
            **{
                widget: key
                for key, widget in panel._value_edits.items()
            },
            panel._criterion: "criterion",
            panel._mode: "search_mode",
            panel._adaptive: "adaptive",
            panel._n_trials: "n_trials",
            panel._n_folds: "n_folds",
            panel._seed: "random_seed",
            panel._resume: "resume_search",
            panel._adaptive_n_step: "n_neighbors_step",
            panel._adaptive_d_step: "min_dist_step",
            panel._adaptive_rounds: "n_trials",
            panel._adaptive_improvement: "min_improvement",
            panel._stability_repeats: "umap_stability_repeats",
            panel._neighborhood_weight: "umap_neighborhood_weight",
            panel._stability_weight: "umap_stability_weight",
            panel._cluster_weight: "umap_cluster_structure_weight",
            panel._max_panels: "max_panels",
        }
        # No API link dots, matching the Live Preview panel and the main
        # settings form -- both of which dropped them for the same reason.
        # A dot after the setting AND after the field doubles the visual
        # noise of a form the user is reading as a list of choices, and it
        # carries a tooltip of its own, so the help pops when the pointer is
        # over the row's right-hand side. From the user's side of the screen
        # that reads as "the field has a tooltip".
        #
        # The linked help itself is untouched: it stays on the LABEL, which
        # is what `tests/qt/test_all_module_smoke.py` asserts (every setting
        # label carries help containing `href=`). Only the dot goes.
        install_api_tooltips(self, panel.app_key, search_tooltips,
                             api_dots=False)
        self.resize(820, 760)

    def _build_umap_tabs(self) -> None:
        """Materialize every UMAP module category as its own settings tab."""
        from .settings_model import SettingsWidgets

        self._module_model = SettingsWidgets("umap", parent=self)
        sections = self._module_model.build_sections()
        for key, value in self._panel._settings.items():
            self._module_model.set_value_for_key(key, value)
        relevant = {
            "Embedding & Clustering", "Plot", "Advanced", "UMAP Display",
        }
        for title, rows in sections:
            if title not in relevant:
                continue
            category_keys = [
                key for key, widget in self._module_model._widgets.items()
                if any(widget is row_widget for _label, row_widget in rows)
            ]
            self._module_keys.update(category_keys)
            page = QWidget()
            page.setObjectName("UmapSettingsPage")
            form = QFormLayout(page)
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            for label, widget in rows:
                form.addRow(label, widget)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.NoFrame)
            scroll.setWidget(page)
            self._tabs.addTab(scroll, title)
        # SettingsWidgets materializes every UMAP category with ``self`` as
        # the initial parent. This popup intentionally shows only the four
        # graph/embedding categories above. Remove controls from omitted
        # categories entirely: merely hiding RowExclusionEditor left its
        # "+ Add exclusion" child eligible for a transient paint at (0, 0),
        # which is the clipped "Ad…sion" text reported over the Search tab.
        for key, widget in list(self._module_model._widgets.items()):
            if key not in self._module_keys:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
                del self._module_model._widgets[key]

    def propagate_settings(self) -> None:
        callback = self._panel._apply_cb
        if callback is None:
            return
        values = {}
        if self._module_model is not None:
            collected = self._module_model.collect()
            values = {
                key: collected[key] for key in self._module_keys
                if key in collected
            }
        # One-value search fields are valid module settings. A selected result
        # is more authoritative and therefore wins.
        for key, _label, kind in APP_PARAMS[self._panel.app_key]:
            try:
                parsed = parse_values(
                    self._panel._value_edits[key].text(), kind, key)
            except ValueError:
                parsed = []
            if len(parsed) == 1:
                values[key] = parsed[0]
        selected = self._panel.selected_params()
        if selected:
            values.update(selected)
        callback(values)
        self._panel._status.setText(
            "Propagated UMAP search and graph settings to the module.")

    def closeEvent(self, event):  # noqa: N802
        self._panel._settings_panel.setParent(self._panel)
        self._panel._settings_panel.hide()
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


class WalkAxesDialog(QDialog):
    """Choose which parameters a Walk searches, and how finely.

    One row per structural UMAP parameter: whether it takes part, where the
    walk starts on it, and the axis resolution -- how many values that axis
    contributes to each round. 2 is the classic pair either side of the
    centre; 3 includes the centre; 5 reaches two steps out.

    The round cost is the PRODUCT of the resolutions, which is why the
    dialog shows that number and says when it will trip the per-round
    limit. Ten axes at resolution 2 is 1024 fits to take one step, and a
    user who has to discover that by waiting has been failed by the
    control, not by the search.
    """

    def __init__(self, panel: "HyperparamPanel"):
        super().__init__(panel)
        self._panel = panel
        self.setObjectName("WalkAxesDialog")
        self.setWindowTitle("Walk axes")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)

        blurb = QLabel(
            "The Walk searches the space these parameters define. Every one "
            "of them changes the structure of the embedding, so searching "
            "only n_neighbors and min_dist leaves the rest at a default "
            "nobody chose.")
        blurb.setWordWrap(True)
        blurb.setObjectName("WalkAxesBlurb")
        outer.addWidget(blurb)

        page = QWidget(self)
        page.setObjectName("WalkAxesPage")
        grid = QGridLayout(page)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        for column, title in enumerate(("Search", "Start at", "Resolution")):
            header = QLabel(title)
            header.setObjectName("WalkAxesHeader")
            grid.addWidget(header, 0, column)

        chosen = dict(panel.walk_axes())
        self._rows: Dict[str, Tuple[Toggle, QWidget, QSpinBox]] = {}
        for row, name in enumerate(UMAP_WALK_PARAMETERS, start=1):
            spec = UMAP_WALK_PARAMETERS[name]
            enable = Toggle(name)
            enable.setChecked(name in chosen)
            grid.addWidget(enable, row, 0)

            start_value = str(chosen.get(name, {}).get("start", ""))
            if not start_value:
                start_value = panel.walk_start_for(name)
            if name == "metric":
                choices = tuple(umap_metrics())
            else:
                choices = spec.get("choices")
            if choices:
                start: QWidget = QComboBox()
                start.addItems([str(c) for c in choices])
                if start_value in choices:
                    start.setCurrentText(start_value)
            else:
                start = QLineEdit(start_value)
                start.setPlaceholderText("one value")
            start.setObjectName(f"WalkStart_{name}")
            grid.addWidget(start, row, 1)

            resolution = QSpinBox()
            resolution.setObjectName(f"WalkResolution_{name}")
            resolution.setRange(2, 7)
            resolution.setValue(int(chosen.get(name, {}).get("resolution", 2)))
            resolution.setToolTip(
                f"How many values of {name} each round scores. 2 is one step "
                "either side of the centre; an odd number keeps the centre "
                "in the round as well.")
            grid.addWidget(resolution, row, 2)

            self._rows[name] = (enable, start, resolution)
            enable.toggled.connect(self._update_cost)
            resolution.valueChanged.connect(self._update_cost)

        grid.setColumnStretch(1, 1)
        scroll = QScrollArea(self)
        scroll.setObjectName("WalkAxesScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(page)
        outer.addWidget(scroll, 1)

        self._cost = QLabel()
        self._cost.setObjectName("WalkAxesCost")
        self._cost.setWordWrap(True)
        outer.addWidget(self._cost)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self._style_surfaces()
        self._update_cost()

    def _style_surfaces(self) -> None:
        """Paint this dialog's containers, locally.

        `WalkAxesPage` is an anonymous QWidget inside a scroll area, and an
        unstyled one inherits the blanket `QWidget { background-color: bg }`
        rule and paints the window colour as a solid rectangle. Styling it
        here rather than through `register_widget_qss` is deliberate: this
        module is imported when a module screen is first built, which is
        after the application stylesheet has been composed.
        """
        palette = active_palette()
        bg = css_color(palette["bg"])
        fg = css_color(palette["fg"])
        muted = css_color(palette.get("muted", palette["fg"]))
        self.setStyleSheet(f"""
            QDialog#WalkAxesDialog,
            QDialog#WalkAxesDialog QWidget#WalkAxesPage,
            QDialog#WalkAxesDialog QScrollArea,
            QDialog#WalkAxesDialog QScrollArea > QWidget > QWidget {{
                background-color: {bg};
                color: {fg};
            }}
            QDialog#WalkAxesDialog QLabel#WalkAxesBlurb,
            QDialog#WalkAxesDialog QLabel#WalkAxesCost {{
                color: {muted};
            }}
            QDialog#WalkAxesDialog QLabel#WalkAxesHeader {{
                color: {muted};
                font-weight: 600;
            }}
        """)

    def _update_cost(self, *_args) -> None:
        """Say what one round will cost, before it is paid.

        The number is taken from the ENGINE -- the same
        :func:`walk_neighbourhood` the search will call -- rather than
        re-derived here. The arithmetic is not "product minus the centre":
        an even resolution leaves the centre out of the axis entirely, a
        categorical axis always includes it, and a value at a boundary
        clamps two offsets onto one. A second implementation of that in the
        dialog would be wrong in a way nobody would notice until a round
        cost something other than what was promised.
        """
        selection = self.selection()
        count = len(selection)
        if count == 0:
            self._cost.setText(
                "No axes selected — the Walk falls back to n_neighbors and "
                "min_dist, its original two.")
            return
        try:
            start = {name: _parse_walk_start(name, spec["start"])
                     for name, spec in selection.items()}
            axes = umap_walk_axes(
                start, parameters=list(selection),
                resolutions={n: s["resolution"]
                             for n, s in selection.items()})
            moves, full = walk_neighbourhood(axes, start)
            fits = len(moves)
        except (ValueError, KeyError):
            # A half-typed starting value. Say nothing rather than a number
            # that is wrong; the OK button validates properly.
            self._cost.setText(
                f"{count} axes. Fill in a starting value for each to see "
                "what a round will cost.")
            return
        if not full:
            self._cost.setText(
                f"{count} axes → {fits} fits per round. The full "
                f"neighbourhood is past the {MAX_WALK_CANDIDATES_PER_ROUND} "
                "limit, so each round varies ONE axis at a time instead: "
                "still enough to pick a direction, but it does not see "
                "interactions between axes.")
        else:
            self._cost.setText(
                f"{count} axes → {fits} fits per round.")

    def selection(self) -> Dict[str, Dict[str, Any]]:
        """The chosen axes as ``{name: {'start': str, 'resolution': int}}``."""
        out: Dict[str, Dict[str, Any]] = {}
        for name, (enable, start, resolution) in self._rows.items():
            if not enable.isChecked():
                continue
            text = (start.currentText() if isinstance(start, QComboBox)
                    else start.text())
            out[name] = {"start": str(text).strip(),
                         "resolution": int(resolution.value())}
        return out
