"""Alternative launcher for spaCR's spaceout visual mode.

The command starts the standard :func:`spacr.qt.run` application with a
contrast-checked spectral palette and fractal ambient animation. The mode is
process-local: it is not stored as a preference and does not alter subsequent
standard launches. Existing animation preferences remain effective, including
the option to disable ambient animation. It is also the only mode with sound:
Preferences shows its Sound tab (and its Fractal tab) only here, and ordinary
spaCR plays nothing whatever the stored sound settings say.
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Launch spaCR with process-local spaceout rendering enabled.

    :param argv: Optional spaCR command-line arguments. ``None`` reads
        ``sys.argv[1:]``.
    :returns: Application exit code from :func:`spacr.qt.run`.
    """
    from . import _prefer_a_context_the_shaders_can_run_on

    _prefer_a_context_the_shaders_can_run_on()

    from . import _install_quiet_qt_logging, _quiet_vispy_logging

    _install_quiet_qt_logging()
    _quiet_vispy_logging()

    from .theme import enable_spaceout
    enable_spaceout()

    from . import run
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
