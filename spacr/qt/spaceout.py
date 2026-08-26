"""The ``spaceout`` command: the same application, wearing something else.

``spaceout`` starts the spaCR the ``spacr`` and ``spacr-qt`` commands start.
Not a fork of it, not a copy of it, and not a reduced version of it: the same
:func:`spacr.qt.run`, the same main window, the same registry of modules, the
same preferences, the same first-run screen. Every screen reachable from
``spacr`` is reachable from here and behaves identically.

What changes is the *dressing*:

* the palette goes rainbow — :func:`spacr.qt.theme.enable_spaceout` re-hues
  whichever theme is resolved, keeping every role's relative luminance, so
  the contrast rules go on passing and the theme contract is untouched;
* the ambient backdrop draws moving fractals instead of drifting blobs —
  :class:`spacr.qt.widgets.ambient.FractalEngine`, chosen by
  :func:`spacr.qt.widgets.ambient.dressed` at the moment a backdrop is built.

It is not a theme menu entry
----------------------------
Nothing in Preferences offers this, and an ordinary ``spacr`` start can never
land in it. That is a property of where the choice is kept rather than of a
control being hidden: it is process state in :mod:`spacr.qt.theme`, set once
by :func:`main` before the application starts and written nowhere. A stored
preference would survive a restart and leak the mode into a normal launch,
which is the one thing the request rules out — so nothing here writes to
:mod:`spacr.qt.preferences`, and the fractal's name is deliberately kept out
of :data:`spacr.qt.widgets.ambient.AMBIENT_THEMES`, which is what the
animation dropdown is built from and what a stored value is validated
against. A settings file that says ``fractal`` is rejected like any other
name this build does not offer.

One thing the dressing does not do is turn the backdrop *on*. A user who set
the Animation preference to None asked for zero frames and zero cost, and a
launcher is not a reason to overrule that; they get the rainbow palette and
no fractals, which is the same application wearing something else rather than
the same application with a setting changed behind their back.
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Launch the GUI in the spaceout dressing.

    The entry point ``spaceout`` (see ``setup.py``), and the whole of it: the
    dressing is switched on and then the ordinary launcher runs. Every
    argument ``spacr`` takes is taken here and means the same thing —
    ``spaceout mask`` opens the Mask screen, ``spaceout --version`` prints the
    version — because the arguments are handed straight to
    :func:`spacr.qt.run`.

    Args:
        argv: Optional CLI arguments. ``None`` reads ``sys.argv[1:]``, which
            is what the console script does.

    Returns:
        Whatever :func:`spacr.qt.run` returns — the ``QApplication.exec()``
        exit code, or 1 if the optional Qt extra is not installed.
    """
    # Before the application: `run` builds the stylesheet from
    # `theme.palette_for`, and the palette has to already be re-hued by then
    # or the first window paints in the undressed colours and only later
    # screens pick the new ones up.
    from .theme import enable_spaceout
    enable_spaceout()

    from . import run
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
