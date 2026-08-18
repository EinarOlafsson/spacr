"""Saved and visible are the SAME event.

Instruction 139 C, reported on 2026-08-18: "all graphs should be sable and
observable in the software, currently several graphs are saved but I cannot
see them in the software".

THE DIAGNOSIS, and it is exact. A figure reaches the GUI by ONE route:
``spacr/qt/bridge.py`` replaces ``matplotlib.pyplot.show`` with a capture that
walks ``plt.get_fignums()`` and emits every figure it finds. So the rule the
application actually runs on is

    a figure is visible if and only if somebody calls plt.show()
    AND the figure is registered with pyplot

`spacr/regression_qc.py` fails BOTH halves, which is why its whole ~19-panel
suite is invisible. It builds bare ``matplotlib.figure.Figure`` objects -- its
own docstring says so, "figures built via matplotlib.figure.Figure are not
registered with pyplot" -- writes them with ``fig.savefig`` and never calls
``show``. Every panel is on disk and none of them is in the application.

That rule was never written down and it punishes the module that behaves best:
building a Figure directly rather than through the global pyplot registry is
the CORRECT thing to do in a library, and it is exactly what made those
figures unreachable.

SO THE DELIVERY IS DECOUPLED FROM PYPLOT. A module hands a figure here; this
saves it AND announces it, and neither depends on a global registry or on a
call that exists for another reason.

THE SINK IS OPTIONAL AND ABSENT BY DEFAULT. A headless run -- `spacr-run`, a
notebook, a test -- installs none, so `publish` writes the file and returns.
Nothing here imports Qt.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

LOG = logging.getLogger(__name__)

#: Where a published figure goes, or None when nobody is listening.
#:
#: Module-level rather than passed down: the code that BUILDS a figure is
#: several calls below the code that knows whether a GUI is attached, and
#: threading a sink through every one of them is how half of them end up not
#: having it -- which is the state this module exists to fix.
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


def clear_sink() -> None:
    """Remove the sink. A run that has finished is not still publishing."""
    set_sink(None)


def sink() -> Optional[Callable[..., Any]]:
    """The installed sink, or None. For a test that wants to assert it."""
    return _sink


def publish(fig, path=None, *, fmt=None, dpi=None, close=False, **kwargs):
    """Save ``fig`` and make it visible. Returns the path written, or None.

    :param fig: a matplotlib Figure. It need NOT be registered with pyplot --
        that is the whole point.
    :param path: where to write it. None publishes without saving, for a
        figure that is worth looking at and not worth keeping.
    :param close: close the figure after publishing. The sink is called
        FIRST, because a closed figure has nothing left to render.

    THE SAVE GOES THROUGH `spacr.plot.save_figure`, so the user's figure
    format and resolution preferences are honoured. A literal '.pdf' here
    would be the complaint this project has already had twice.

    A SINK THAT RAISES DOES NOT LOSE THE FILE. The file is written first and
    the announcement is best-effort: a GUI that has gone away must not take
    the run's output with it.

    A FIGURE THAT WAS NEVER DRAWN IS NOT A FIGURE. ``fig=None`` writes
    nothing, announces nothing and returns None, because the panels that come
    back optional are exactly the ones a caller forgets to check: `ml_analysis`
    returns ``feature_importance_fig = None`` for every model without
    ``feature_importances_`` -- logistic regression and
    HistGradientBoostingClassifier among the offered ones -- and the call site
    handed it straight to ``savefig``, which took a whole scoring run down
    after the model had been fitted and every object scored. Losing a run over
    a picture that was never drawn is the worst trade in this file.
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


__all__ = ["clear_sink", "publish", "set_sink", "sink"]
