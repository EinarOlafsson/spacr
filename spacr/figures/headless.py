"""Render pyqtgraph figures without an interactive window.

Regression plots are rendered with pyqtgraph in the graphical interface. This
module renders the same plot specification through the same renderer in an
offscreen widget, allowing interactive and pipeline-generated figures to
share one implementation.

:func:`spacr.figures.scene.pyqtgraph_ready` supplies a ``QApplication`` on
Qt's ``offscreen`` platform when no display is available. If Qt cannot create
a scene safely, rendering returns an explanatory refusal instead of reporting
a nonexistent output. Output format and resolution follow
:func:`spacr.plot.figure_output_preferences`, and completed files are
published through :func:`spacr.figure_sink.publish_file` for inclusion in the
run gallery.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

LOG = logging.getLogger("spacr.figures.headless")

#: Pixel size the offscreen widget is laid out at before it is exported.
#:
#: BIG ENOUGH THAT THE AXES ARE NOT CRAMPED. The vector export scales from
#: the scene, so this is not the resolution of the file -- but tick spacing,
#: label elision and legend layout are all decided at this size, and a plot
#: laid out at 200 px wide exports its cramped decisions faithfully.
RENDER_SIZE: Tuple[int, int] = (1400, 900)

#: What to say when there is no Qt at all. A run that silently stops
#: writing figures is the worst outcome, so the refusal is loud and names
#: the fix.
NO_QT = ("PySide6 is not importable, so the pyqtgraph figures cannot be "
         "rendered. Install it with `pip install spacr` (it is a core "
         "dependency) or run with `regression_qc=False` to skip them.")

#: What to say when Qt is here but no platform plugin will start.
NO_PLATFORM = ("Qt could not start any window platform, not even offscreen, "
               "so the pyqtgraph figures cannot be rendered. Set "
               "QT_QPA_PLATFORM=offscreen, or install the system Qt "
               "libraries (libgl1, libegl1, libxkbcommon0).")


def application():
    """The ``QApplication`` to render under, or ``(None, reason)``.

    DELEGATED TO :func:`spacr.figures.scene.pyqtgraph_ready`, which is the
    one place that answers "can a scene be built here and now". It knows two
    things this module must not get wrong on its own:

    * pyqtgraph binds its Qt library on FIRST import, and this environment
      also has PyQt6 -- so a bare ``import pyqtgraph`` can leave PySide6
      unloadable. It sets ``PYQTGRAPH_QT_LIB`` before anything imports it.
    * a QWidget must be built on the GUI thread. The regression QC suite runs
      on the run's worker thread under a live application, and a widget built
      there lives on a thread that is about to end -- traced from two
      segfaults that landed nowhere near the cause.

    A second answer to that question is how one of those two rules gets
    forgotten, so there is only one.

    :returns: ``(app, "")`` on success, ``(None, reason)`` when rendering is
        impossible. The reason is a sentence for the user, not a traceback.
    """
    from .scene import pyqtgraph_ready

    ok, reason = pyqtgraph_ready()
    if not ok:
        return None, reason or NO_PLATFORM
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:                       # ready() usually made it
        return None, NO_PLATFORM
    return app, ""


def render_offscreen(spec, path: str, *, size: Optional[Tuple[int, int]] = None,
           fmt: Optional[str] = None, title: str = "",
           x_label: str = "", y_label: str = "",
           publish: bool = True) -> Optional[str]:
    """Draw ``spec`` offscreen and write it to ``path``.

    :param spec: a :class:`spacr.qt.widgets.grouped_plot.PlotSpec`.
    :param path: destination. The extension is rewritten to the user's
        figure-format preference unless ``fmt`` names one.
    :param size: layout size in pixels; :data:`RENDER_SIZE` by default.
    :param fmt: ``"pdf"``, ``"png"`` or ``"svg"`` to override the preference.
    :param title: plot title. ``spec.title`` is used when empty.
    :param x_label: horizontal axis label. ``spec.x_label`` when empty.
    :param y_label: vertical axis label. ``spec.y_label`` when empty.
    :param publish: announce the file to the figure sink. True by default,
        because saved and visible are the same event.
    :returns: the path written, or None with the reason logged at WARNING
        when there is no Qt to render under.
    """
    app, refusal = application()
    if app is None:
        LOG.warning("%s", refusal)
        return None

    from ..plot import figure_output_preferences
    from ..qt.widgets.grouped_plot import GroupedPlot

    chosen = str(fmt or figure_output_preferences()[0]).lower().lstrip(".")
    stem, _ = os.path.splitext(str(path))
    target = f"{stem}.{chosen}"
    parent = os.path.dirname(os.path.abspath(target))
    os.makedirs(parent, exist_ok=True)

    plot = GroupedPlot(title=title or getattr(spec, "title", "") or "",
                       x_label=x_label or getattr(spec, "x_label", "") or "",
                       y_label=y_label or getattr(spec, "y_label", "") or "")
    try:
        plot.resize(*(size or RENDER_SIZE))
        if not plot.show_spec(spec):
            LOG.debug("nothing to draw for %s", target)
            return None
        # PROCESS EVENTS BEFORE EXPORTING. Offscreen, `resize` posts a layout
        # that nothing has delivered yet, so an export taken straight after
        # it photographs the widget's startup geometry -- which is how a
        # figure comes out with its axes in the wrong place.
        app.processEvents()
        written = plot.export(target)
    finally:
        plot.deleteLater()

    if written and publish:
        from ..figure_sink import publish_file

        publish_file(written, title or getattr(spec, "title", "") or None)
    return written


def render_bundle(spec, folder: str, name: str, **kwargs) -> Optional[str]:
    """Render ``spec`` and write the whole folder beside it.

    The same bundle :meth:`FastPlot.export_bundle` writes from the screen --
    figure, data, statistics, settings -- produced by a run that has no
    screen, so a generated figure and a saved one are the same thing.

    :param spec: a :class:`spacr.qt.widgets.grouped_plot.PlotSpec`.
    :param folder: parent directory for the bundle.
    :param name: bundle name; also the figure's name inside it.
    :returns: the bundle directory, or None when there is no Qt.
    """
    app, refusal = application()
    if app is None:
        LOG.warning("%s", refusal)
        return None

    from ..qt.widgets.grouped_plot import GroupedPlot

    plot = GroupedPlot(title=kwargs.pop("title", "") or name,
                       x_label=kwargs.pop("x_label", "") or "",
                       y_label=kwargs.pop("y_label", "") or "")
    try:
        plot.resize(*kwargs.pop("size", RENDER_SIZE))
        if not plot.show_spec(spec):
            return None
        app.processEvents()
        return plot.export_bundle(folder, name)
    finally:
        plot.deleteLater()
