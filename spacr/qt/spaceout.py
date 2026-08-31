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


def _prefer_a_context_the_shaders_can_run_on() -> None:
    """Ask for XWayland when the session is Wayland, before Qt starts.

    MEASURED, and confirmed by the person running it. On a native Wayland
    session Qt hands vispy an OpenGL ES context; vispy compiles the fractal
    shaders as desktop GLSL 120, and ES answers "unsupported version 120",
    so the backdrop never draws a frame. The identical code under ``xcb``
    gets a GLX context and draws with no errors at all.

    Nothing in spaCR changed when this started happening -- the session
    did. So this is not a workaround for a bug in the backdrop; it is
    asking for the context the backdrop has always needed.

    Only when the caller has expressed no preference of their own. An
    explicit QT_QPA_PLATFORM is always honoured, including a deliberate
    ``wayland`` by someone who would rather have no backdrop than
    XWayland, and the variable is left alone when there is no X server to
    fall back to.
    """
    import os

    if os.environ.get("QT_QPA_PLATFORM"):
        return                      # the caller chose; do not overrule them
    if not (os.environ.get("WAYLAND_DISPLAY")
            or os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"):
        return                      # not Wayland; the context is already fine
    if not os.environ.get("DISPLAY"):
        return                      # no XWayland to ask for
    os.environ["QT_QPA_PLATFORM"] = "xcb"
