"""Render generated regression figures from their interactive scenes.

For the plots listed in :data:`FAST_PANELS`, :func:`render_panel` exports a
provided ``FastPlot`` directly so the saved figure matches the visible tab.
In ``auto`` mode, a call without a live widget uses the corresponding
matplotlib panel. ``SPACR_FIGURE_RENDERER=pyqtgraph`` explicitly requests a
new scene built from the coefficient table; ``matplotlib`` forces that
renderer.

Output paths pass through :func:`spacr.plot.figure_path`, so file extensions
follow the configured format. Matplotlib figures are published through
``spacr.figure_sink.publish`` and scene exports through
``spacr.figure_sink.publish_file``; both routes add saved files to the gallery
when a listener is present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence

from .panels import SHEET_ORDER

#: The seven duplicated plots: house-style panel key -> the class in
#: ``spacr.qt.widgets.fast_plots`` that draws the same picture on screen.
#:
#: This mapping IS the claim that the two are the same plot. A key here whose
#: two renderers show different things is the bug this module exists to
#: remove, so a test asserts the keys are exactly ``SHEET_ORDER``.
FAST_PANELS = {
    "volcano": "VolcanoPlot",
    "effect_rank": "EffectRankPlot",
    "effect_distribution": "EffectDistribution",
    "controls": "ControlSeparation",
    "agreement": "GuideAgreementPlot",
    "p_histogram": "PValueHistogram",
    "qq": "QQPlot",
}

#: What a renderer choice can be. ``auto`` is the rule the module docstring
#: states; the other two are a person overruling it.
RENDERERS = ("auto", "pyqtgraph", "matplotlib")

#: A QApplication this module had to start, kept alive. Qt destroys the
#: application the moment its last Python reference goes, and an export that
#: happens after that is a segfault rather than an exception.
_APPLICATION = None


@dataclass
class RenderedPanel:
    """One generated figure: what drew it, where it went, and why.

    ``renderer`` is RECORDED rather than assumed, because the answer varies
    per run and per machine and a user comparing a figure on screen against a
    figure in a folder has to be able to find out which one they are holding.

    :ivar key: generated-panel key identifying the requested figure.
    """

    key: str
    path: Optional[str] = None
    renderer: str = ""
    drawn: bool = False
    reason: str = ""

    def __bool__(self) -> bool:
        """Return whether the panel was drawn and has an output path.

        :returns: ``True`` only for a drawn panel with a non-empty path.
        """
        return bool(self.drawn and self.path)


def requested_renderer() -> str:
    """The renderer the environment asks for, or ``'auto'``.

    An unrecognised value is ``'auto'`` rather than an error: a run must not
    lose its figures over a misspelt environment variable, which is the rule
    :func:`spacr.figure_style.figure_save_mode` already follows for the save
    mode.
    """
    asked = os.environ.get("SPACR_FIGURE_RENDERER", "").strip().lower()
    return asked if asked in RENDERERS else "auto"


def qt_application():
    """The running ``QApplication``, or None. NEVER creates one.

    Deliberately does not import PySide6 unless it is already imported.
    Importing Qt pulls a GUI toolkit into a notebook that asked for a
    regression, so the question "is there a GUI?" has to be answerable
    without answering it in the affirmative by accident.
    """
    import sys

    module = sys.modules.get("PySide6.QtWidgets")
    if module is None:
        return None
    try:
        return module.QApplication.instance()
    except Exception:                                          # noqa: BLE001
        return None


# THERE IS NO DETECTION HERE, AND THAT IS THE DESIGN. Two attempts at asking
# "is the GUI up?" were built and both were wrong, so the question is not
# asked at all: a scene is rendered when a caller HANDS ONE IN, and otherwise
# the page that has always been written is written.
#
#   * "a QApplication exists" is false. Measured 2026-08-18: matplotlib's
#     QtAgg backend -- the DEFAULT backend in this environment -- calls
#     `_create_qApp` from inside `plt.figure()` and constructs a
#     `QApplication(["matplotlib"])`. So the first matplotlib panel of a
#     headless run created one, and every panel after it saw a live
#     QApplication and switched renderer. One run, seven figures, two
#     libraries: a worse disagreement than the one this module removes.
#   * "`spacr.qt.widgets.fast_plots` is in sys.modules" is also false. A test
#     that does nothing but check the seven classes exist puts it there, and
#     so does any import of the widget package. Module presence is not
#     evidence that a plot was ever built, let alone that one is on screen.
#
# The unambiguous fact is the widget itself. `render_panel(..., plot=widget)`
# renders that widget; nothing else can be mistaken for it.


def renderer_for(key: str, force: Optional[str] = None) -> tuple:
    """``(renderer, reason)`` for one panel. The decision, in one place.

    :param key: generated-panel key. Only keys in :data:`FAST_PANELS` have an
        interactive twin; every other key is assigned to matplotlib.
    :param force: one of :data:`RENDERERS`, overriding the environment and
        the auto rule.
    :returns: ``('pyqtgraph'|'matplotlib', reason)``. The reason is never
        empty for matplotlib, because "why is this not the screen's renderer"
        is exactly the question a user asks of a figure that does not match a
        tab.
    """
    choice = str(force).strip().lower() if force else requested_renderer()
    if choice not in RENDERERS:
        choice = "auto"
    if key not in FAST_PANELS:
        return "matplotlib", f"{key!r} has no interactive twin"
    if choice == "matplotlib":
        return "matplotlib", "matplotlib was asked for"
    if choice == "auto":
        return "matplotlib", ("no live plot was handed in, so there is no "
                              "scene to render and no tab to disagree with")
    available, why = _pyqtgraph_ready(create=True)
    if not available:
        return "matplotlib", why
    return "pyqtgraph", ""


def _pyqtgraph_ready(create: bool = True) -> tuple:
    """``(ok, reason)``: can a scene be BUILT here and now?

    Reached only when pyqtgraph was explicitly asked for, which is what
    licenses starting a ``QApplication``: importing Qt costs ~1 s and pulls a
    GUI toolkit into a notebook, so it happens on request and never on a
    guess.
    """
    global _APPLICATION

    application = qt_application()
    # PySide6 IS NAMED BEFORE pyqtgraph IS IMPORTED, AND THE ORDER IS
    # LOAD-BEARING. pyqtgraph binds to the first of PyQt5, PyQt6, PySide2,
    # PySide6 that imports, and PyQt6 is installed in this environment -- so
    # `import pyqtgraph` first loads PyQt6's libQt6Core and PySide6 6.11 then
    # cannot load at all: "libpyside6.abi3.so.6.11: undefined symbol:
    # _ZN9QtPrivate9sizedFreeEPvm". Measured 2026-08-18 on the generated-figure
    # path, where it silently sent a whole QC suite back to matplotlib.
    os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
    try:
        from ..qt.widgets.fast_plots import HAVE_PYQTGRAPH
    except Exception as error:                                 # noqa: BLE001
        return False, f"pyqtgraph plots are unavailable: {error}"
    if not HAVE_PYQTGRAPH:
        return False, "pyqtgraph is not installed"
    if application is None:
        try:
            from PySide6.QtWidgets import QApplication

            if not os.environ.get("DISPLAY") and not os.environ.get(
                    "WAYLAND_DISPLAY"):
                # MEASURED, not assumed: with this set, `ImageExporter` writes
                # a PNG and `QPdfWriter` + `scene().render()` writes a real
                # vector PDF on a machine with no display at all.
                os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            _APPLICATION = QApplication.instance() or QApplication([])
        except Exception as error:                             # noqa: BLE001
            return False, f"no QApplication could be started: {error}"
    return True, ""


# --------------------------------------------------------------------------- #
#  Feeding a scene from a coefficient table
# --------------------------------------------------------------------------- #

def build_fast_plot(key: str, frame, *, alpha: float = 0.05):
    """A live ``FastPlot`` for ``key``, fed from ``frame``.

    THE COLUMNS ARE RESOLVED BY :mod:`spacr.figures.panels`, not here. A
    second opinion about which column is the effect and which rows are
    hypotheses is precisely the disagreement this module exists to end, and
    ``effect_column`` / ``p_column`` / ``q_column`` / ``tested`` are the
    generated side's single statement of it.

    :param key: generated-panel key with an entry in :data:`FAST_PANELS`.
    :param frame: coefficient table used to populate the interactive plot.
    :returns: the widget, or None when this table cannot support the panel.
    :raises KeyError: on a key with no interactive twin.
    """
    from ..qt.widgets import fast_plots
    from .panels import effect_column, p_column, q_column, tested

    if key not in FAST_PANELS:
        raise KeyError(f"{key!r} has no pyqtgraph twin; known: "
                       f"{', '.join(sorted(FAST_PANELS))}")
    if frame is None or not len(frame):
        return None
    effect = effect_column(frame)
    raw_p = p_column(frame)
    keys = (frame["feature"].astype(str).tolist()
            if "feature" in frame.columns else None)
    plot = getattr(fast_plots, FAST_PANELS[key])()

    if key == "volcano":
        if effect is None or raw_p is None:
            return None
        plot.set_results(frame, effect=effect, p_column=raw_p, alpha=alpha,
                         q_column=q_column(frame),
                         key_column="feature" if keys else None)
    elif key == "effect_rank":
        if effect is None:
            return None
        plot.set_results(frame, effect=effect, alpha=alpha,
                         key_column="feature" if keys else None)
    elif key == "effect_distribution":
        if effect is None:
            return None
        # THE FAMILY, NOT THE AXIS. The house-style panel histograms the
        # TESTED coefficients; handing this one the whole table would put the
        # intercept and the nuisance terms into a picture whose caption says
        # "the tested coefficients", which is the two renderers disagreeing
        # about what the figure is OF.
        rows = tested(frame)
        plot.set_effects(frame.loc[rows, effect],
                         keys=_subset(keys, rows),
                         untested=int(len(frame) - int(rows.sum())))
    elif key in ("p_histogram", "qq"):
        if raw_p is None:
            return None
        rows = tested(frame)
        plot.set_p_values(frame.loc[rows, raw_p], keys=_subset(keys, rows))
    elif key == "controls":
        groups, group_keys = _control_groups(frame, effect)
        if len(groups) < 2:
            return None
        plot.set_groups(groups, keys=group_keys or None)
    else:
        # FAST_PANELS is exhaustive; this is the former
        # ``elif key == "agreement":`` arm after the six keys above.
        from ..guide_concordance import guide_support

        if effect is None or "feature" not in frame.columns:
            return None
        support = guide_support(frame, alpha=alpha)
        if support is None or not len(support):
            return None
        plot.set_support(support,
                         keys=(support["feature"]
                               if "feature" in support.columns else None))
    return plot


def _subset(keys, rows):
    """``keys`` restricted to ``rows``, or None when there were none."""
    if keys is None:
        return None
    return [key for key, keep in zip(keys, rows) if keep]


def _control_groups(frame, effect) -> tuple:
    """``({label: values}, {label: keys})`` split by the fit's own labels.

    The label names match :func:`spacr.figures.panels.control_separation`
    exactly -- ``nc`` is "negative", ``pc`` is "positive" -- because a control
    called "negative" on screen and "nc" in the paper figure is the same
    disagreement in a smaller place.
    """
    from .panels import _column

    condition = _column(frame, "condition", "control", "class")
    if effect is None or condition is None:
        return {}, {}
    names = {"nc": "negative", "pc": "positive", "control": "control",
             "other": "screen"}
    groups, keys = {}, {}
    for value, label in names.items():
        rows = frame[frame[condition].astype(str) == value]
        if len(rows):
            groups[label] = rows[effect].to_numpy()
            if "feature" in rows.columns:
                keys[label] = rows["feature"].astype(str).tolist()
    return groups, keys


# --------------------------------------------------------------------------- #
#  Writing one
# --------------------------------------------------------------------------- #

def render_panel(key: str, frame=None, path=None, *, plot=None,
                 fmt: Optional[str] = None, renderer: Optional[str] = None,
                 alpha: float = 0.05, announce: bool = True) -> RenderedPanel:
    """Write one generated panel, from the scene where there is one.

    :param key: a key of :data:`FAST_PANELS` (equivalently of
        :data:`spacr.figures.panels.SHEET_ORDER`).
    :param frame: the coefficient table. Needed only when no ``plot`` is given
        and for the matplotlib fallback.
    :param path: destination. Its extension is REPLACED by the figure-format
        preference unless ``fmt`` forces one, so the name always names what
        was actually written.
    :param plot: a live ``FastPlot`` to render. THIS IS THE POINT OF THE
        MODULE: given the widget the user is looking at, the file IS that
        widget rather than a second drawing of its data.
    :param renderer: force one of :data:`RENDERERS`.
    :param announce: put the file in the gallery as well as on disk
       .
    :returns: a :class:`RenderedPanel`, always. A panel that could not be
        drawn comes back with ``drawn=False`` and a reason rather than
        raising -- losing a fit over a picture is the worst trade here.
    """
    from ..plot import figure_path

    chosen, why = renderer_for(key, renderer)
    if plot is not None:
        # A live widget IS the pyqtgraph answer. Being handed one and then
        # drawing matplotlib because no QApplication was detected would be
        # absurd: the widget could not exist without one.
        chosen, why = "pyqtgraph", ""
    destination = figure_path(path, fmt) if path else None

    if chosen == "pyqtgraph":
        rendered = _render_with_pyqtgraph(key, frame, destination, plot=plot,
                                          alpha=alpha, announce=announce)
        if rendered.drawn or plot is not None:
            return rendered
        # A scene that could not be built is not a reason to write nothing.
        why = rendered.reason or "the scene could not be built"

    return _render_with_matplotlib(key, frame, path, fmt=fmt, reason=why,
                                   announce=announce)


def _render_with_pyqtgraph(key, frame, destination, *, plot=None,
                           alpha=0.05, announce=True) -> RenderedPanel:
    """Render the scene. Never raises; a failure comes back as a reason."""
    owned = plot is None
    try:
        if owned:
            plot = build_fast_plot(key, frame, alpha=alpha)
        if plot is None:
            return RenderedPanel(key, renderer="pyqtgraph", drawn=False,
                                 reason="this table cannot support the panel")
        if destination is None:
            return RenderedPanel(key, renderer="pyqtgraph", drawn=False,
                                 reason="no destination was given")
        folder = os.path.dirname(str(destination))
        if folder:
            os.makedirs(folder, exist_ok=True)
        written = plot.export(destination) or destination
    except Exception as error:                                 # noqa: BLE001
        return RenderedPanel(key, renderer="pyqtgraph", drawn=False,
                             reason=f"{type(error).__name__}: {error}")
    finally:
        if owned and plot is not None:
            try:
                plot.deleteLater()
            except Exception:                                  # noqa: BLE001
                pass
    if announce:
        from ..figure_sink import publish_file

        publish_file(written, title=key)
    return RenderedPanel(key, path=str(written), renderer="pyqtgraph",
                         drawn=True)


def _render_with_matplotlib(key, frame, path, *, fmt=None, reason="",
                            announce=True) -> RenderedPanel:
    """The house-style panel, published exactly as it always was."""
    import matplotlib.pyplot as plt

    from ..figure_sink import publish
    from .sheet import build_panel

    if frame is None or not len(frame):
        return RenderedPanel(key, renderer="matplotlib", drawn=False,
                             reason=reason or "no coefficients")
    try:
        figure, panel = build_panel(key, frame)
    except Exception as error:                                 # noqa: BLE001
        return RenderedPanel(key, renderer="matplotlib", drawn=False,
                             reason=f"{type(error).__name__}: {error}")
    if not panel.drawn:
        plt.close(figure)
        return RenderedPanel(key, renderer="matplotlib", drawn=False,
                             reason=panel.reason or reason)
    try:
        if announce:
            written = publish(figure, path, fmt=fmt, bbox_inches="tight")
        else:
            from ..plot import save_figure

            written = (save_figure(figure, path, fmt=fmt, bbox_inches="tight")
                       if path is not None else None)
    finally:
        # `build_panel` goes through `plt.figure`, so pyplot holds a
        # reference and `clf()` would clear the figure without releasing it.
        # Seven panels a run, leaked, is how a long session runs out of
        # memory drawing pictures nobody is looking at.
        plt.close(figure)
    return RenderedPanel(key, path=(str(written) if written else None),
                         renderer="matplotlib", drawn=True, reason=reason)


def write_panels(frame, dst, *, keys: Sequence[str] = SHEET_ORDER,
                 plots=None, fmt: Optional[str] = None,
                 renderer: Optional[str] = None, alpha: float = 0.05,
                 verbose: bool = True) -> list:
    """Write every house-style panel into ``dst``. Returns the records.

    :param frame: coefficient/results table used to build panels that have no
        live plot and by any matplotlib fallback.
    :param dst: output directory, created when absent; each panel key becomes
        the destination file stem within it.
    :param plots: ``{key: live FastPlot}`` for the panels that are on screen.
        Anything absent is built from the frame.
    :param verbose: print one line naming the renderer that drew them, so a
        user who finds a figure that does not match a tab can see why in the
        run's log rather than by inspecting the file.
    """
    plots = dict(plots or {})
    folder = str(dst)
    os.makedirs(folder, exist_ok=True)
    # ONE RENDERER FOR THE WHOLE SET, DECIDED ONCE. Asking per panel is not
    # the same question asked seven times -- see the comment above
    # :func:`renderer_for`, where an earlier per-panel rule drew one run's
    # first figure in matplotlib and its other six in pyqtgraph.
    chosen = renderer
    if chosen is None:
        chosen = ("pyqtgraph" if plots
                  else renderer_for(keys[0] if keys else "volcano")[0])
    records = [render_panel(key, frame, os.path.join(folder, key),
                            plot=plots.get(key), fmt=fmt, renderer=chosen,
                            alpha=alpha)
               for key in keys]
    if verbose:
        counts: dict = {}
        for record in records:
            if record.drawn:
                counts[record.renderer] = counts.get(record.renderer, 0) + 1
        drawn = sum(counts.values())
        summary = ", ".join(f"{count} by {name}"
                            for name, count in sorted(counts.items()))
        print(f"[figures] {drawn}/{len(records)} regression panel(s) written "
              f"to {folder}" + (f" ({summary})" if summary else ""))
        for record in records:
            if not record.drawn and record.reason:
                print(f"[figures]   {record.key} not drawn: {record.reason}")
    return records


__all__ = ["FAST_PANELS", "RENDERERS", "RenderedPanel", "build_fast_plot",
           "qt_application", "render_panel", "renderer_for",
           "requested_renderer", "write_panels"]
