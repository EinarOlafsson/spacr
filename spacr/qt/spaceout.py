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
    _prefer_a_context_the_shaders_compile_on()

    from . import _install_quiet_qt_logging, _quiet_vispy_logging

    _install_quiet_qt_logging()
    _quiet_vispy_logging()

    from .theme import enable_spaceout
    enable_spaceout()

    from . import run
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())


def _prefer_a_context_the_shaders_compile_on() -> None:
    """Ask for XWayland when the session is Wayland, before Qt starts.

    MEASURED, not assumed. On this Wayland session Qt hands vispy a GLES
    context; vispy prepends ``#version 120`` -- desktop GLSL -- and the
    fractal vertex shader will not compile, so the GPU backdrop is refused
    and the CPU renderer takes over. That renderer costs about twenty cores
    at 800x600: 4.0 s of wall clock burned 82 s of CPU, which starves the
    GUI thread, so the window cannot be dragged, the app and the preferences
    dialog are slow to open, and the Mandelbrot pattern dumps core. The
    identical session under ``xcb`` gets a GLX context, runs the same
    backdrop on the GPU, and costs 1% of one core.

    So this is not a cosmetic preference: it is the difference between a
    backdrop that costs nothing and one that makes the application
    unusable. Set only when the caller has expressed no preference of their
    own -- an explicit QT_QPA_PLATFORM is always honoured, including a
    deliberate ``wayland``.
    """
    import os

    if os.environ.get("QT_QPA_PLATFORM"):
        return                      # the caller chose; do not overrule them
    if not (os.environ.get("WAYLAND_DISPLAY")
            or os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"):
        return                      # not a Wayland session; nothing to avoid
    if not os.environ.get("DISPLAY"):
        return                      # no XWayland to fall back to
    os.environ["QT_QPA_PLATFORM"] = "xcb"
