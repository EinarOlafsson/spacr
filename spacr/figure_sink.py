"""Save figures and optionally publish them to an attached viewer.

Publishing is independent of Matplotlib's global pyplot registry, so figures
constructed with :class:`matplotlib.figure.Figure` are handled the same way as
pyplot figures. Headless callers install no sink; files are still written and
this module does not import Qt.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

LOG = logging.getLogger(__name__)

#: Callback that receives published figures, or ``None`` when detached.
_sink: Optional[Callable[..., Any]] = None


def set_sink(sink: Optional[Callable[..., Any]]):
    """Install the callable that receives published figures.

    :param sink: called ``sink(fig, path)`` on the thread that published. It
        must not touch a widget -- the GUI side renders to a PNG and hands
        that over a signal.
    :returns: the sink that was installed before, so a caller can put it back.
    """
    global _sink
    previous, _sink = _sink, sink
    return previous


#: Where a published FILE goes, or None when nobody is listening.
#:
#: Separate from :data:`_sink` because the two carry different things and a
#: consumer cannot fake one from the other. A matplotlib sink is handed a live
#: Figure and renders it; a file sink is handed a path to a picture that was
#: already written by something that is not matplotlib, such as a pyqtgraph
#: scene, and there is no Figure to hand it. Reusing the
#: figure sink with ``fig=None`` would hand every existing consumer a None it
#: has no reason to expect.
_file_sink: Optional[Callable[..., Any]] = None


def set_file_sink(sink: Optional[Callable[..., Any]]):
    """Install the callable that receives published FILES.

    :param sink: called ``sink(path, title)`` on the thread that published.
        Like :func:`set_sink` it must not touch a widget.
    :returns: the sink that was installed before, so a caller can put it back.
    """
    global _file_sink
    previous, _file_sink = _file_sink, sink
    return previous


def file_sink() -> Optional[Callable[..., Any]]:
    """The installed file sink, or None. For a test that wants to assert it."""
    return _file_sink


def publish_file(path, title=None):
    """Announce a figure FILE somebody else already wrote. Returns the path.

    :param path: existing figure file to announce to the active sink.

    The rule -- saved and visible are the same event -- with
    the half that :func:`publish` cannot cover. A pyqtgraph scene exported by
    ``FastPlot.export`` is a finished file and never was a matplotlib Figure,
    so there is nothing for the figure sink to render; without this, moving a
    generated plot to the screen's renderer would silently take it out of the
    gallery, which is the exact bug 139 C was filed for.

    A SINK THAT RAISES DOES NOT LOSE THE FILE, for the same reason as in
    :func:`publish`: the file is already on disk and the announcement is
    best-effort, so a GUI that has gone away must not take the run's output
    with it.
    """
    if not path:
        return None
    if _file_sink is not None:
        try:
            _file_sink(str(path), title)
        except Exception:                                      # noqa: BLE001
            LOG.debug("a file sink refused a figure", exc_info=True)
    return str(path)


def clear_sink() -> None:
    """Remove BOTH sinks. A run that has finished is not still publishing.

    Both, because there are two routes into the gallery now and a run that
    left one of them installed would keep announcing into a screen that has
    moved on -- which is worse than the missing tile it was added to fix.
    """
    set_sink(None)
    set_file_sink(None)


def sink() -> Optional[Callable[..., Any]]:
    """The installed sink, or None. For a test that wants to assert it."""
    return _sink


def publish(fig, path=None, *, fmt=None, dpi=None, close=False, **kwargs):
    """Save ``fig`` and send it to the installed display sink.

    :param fig: Matplotlib figure. ``None`` is accepted and returns ``None``.
    :param path: output path. Omit it to publish without saving.
    :param fmt: explicit output format; otherwise inferred by
        :func:`spacr.plot.save_figure`.
    :param dpi: explicit output resolution; otherwise use the figure setting.
    :param close: close the figure after the sink has received it.
    :returns: path written, or ``None`` when no file was requested or no
        figure was supplied.

    Saving occurs before the best-effort sink notification, so a display-sink
    error cannot remove an output file that was written successfully.
    """
    if fig is None:
        return None
    written = None
    if path is not None:
        from .plot import save_figure

        written = save_figure(fig, path, fmt=fmt, dpi=dpi, close=False,
                              **kwargs)
    if _sink is not None:
        try:
            _sink(fig, written)
        except Exception:                                      # noqa: BLE001
            LOG.debug("a figure sink refused a figure", exc_info=True)
    if close:
        try:
            fig.clf()
        except Exception:                                      # noqa: BLE001
            pass
    return written


__all__ = ["clear_sink", "file_sink", "publish", "publish_file",
           "set_file_sink", "set_sink", "sink"]
