"""Alternative launcher for spaCR's spaceout visual mode.

The command starts the standard :func:`spacr.qt.run` application with a
contrast-checked spectral palette and fractal ambient animation. The mode is
process-local: it is not stored as a preference and does not alter subsequent
standard launches. Existing animation preferences remain effective, including
the option to disable ambient animation.
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Launch spaCR with process-local spaceout rendering enabled.

    :param argv: Optional spaCR command-line arguments. ``None`` reads
        ``sys.argv[1:]``.
    :returns: Application exit code from :func:`spacr.qt.run`.
    """
    # Before the application: `run` builds the stylesheet from
    # `theme.palette_for`, and the palette has to already be re-hued by then
    # or the first window paints in the undressed colours and only later
    # screens pick the new ones up.
    # BEFORE THE THEME, which resolves fonts: Qt reports "OpenType support
    # missing for Open Sans" while a face is being loaded, and `run` does not
    # install the filter until later -- so the warnings that leaked were the
    # ones emitted on the way in.
    from . import _install_quiet_qt_logging, _quiet_vispy_logging

    _install_quiet_qt_logging()
    _quiet_vispy_logging()

    from .theme import enable_spaceout
    enable_spaceout()

    from . import run
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
