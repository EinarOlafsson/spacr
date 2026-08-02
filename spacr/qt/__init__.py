"""
Modern PySide6 (Qt 6) GUI for spacr.

Runs alongside the classic Tk GUI (spacr.gui) — nothing here touches the
Tk stack. To launch:

    spacr-qt            # CLI shortcut (see setup.py entry_points)
    python -m spacr.qt  # equivalent

The old GUI keeps working via:
    spacr               # Tk (classic)
    python -m spacr

The Qt code lives in three layers:
    theme.py              — palette + QSS stylesheet
    widgets/              — reusable custom widgets (tiles, sections, ...)
    screens/              — one Qt widget per app screen
                            (startup, mask, measure, ...)
    app.py                — main window + QApplication bootstrap
"""
from __future__ import annotations

import sys

__all__ = ["run"]

_VERSION_FLAGS = frozenset({"-v", "-version", "--version"})

#: Distributions that only `pip install "spacr[qt]"` brings in. PySide6 is
#: declared in the `qt` extra (setup.py), *not* in core, so a plain
#: `pip install spacr` followed by `spacr` used to die on an unhandled
#: `ImportError: No module named 'PySide6'` raised six frames deep inside
#: `spacr/qt/app.py`. shiboken6 is PySide6's binding runtime and fails the
#: same way when a wheel is half-installed.
_QT_EXTRA_MODULES = frozenset({"PySide6", "shiboken6", "qtawesome"})

_QT_MISSING_MESSAGE = """\
spaCR's graphical interface needs the optional Qt extra, which is not
installed in this environment (missing module: {module}).

Install it with:

    python -m pip install "spacr[qt]"

Then run `spacr` again.

No display available? The pipelines run headless without Qt:

    spacr-run --list\
"""


def _missing_qt_extra(exc: ImportError) -> str | None:
    """Identify the Qt-extra distribution whose absence raised ``exc``.

    Only failures that name a module from :data:`_QT_EXTRA_MODULES` count.
    Anything else is a genuine bug inside the GUI package and must keep its
    traceback rather than be reported as a missing install.

    Args:
        exc: The ``ImportError`` raised while importing :mod:`spacr.qt.app`.

    Returns:
        The top-level module name to name in the install hint, or ``None``
        when ``exc`` is unrelated to the Qt extra.
    """
    # ModuleNotFoundError sets `.name` to the module that could not be found;
    # a failed `from PySide6.QtCore import ...` sets it to `PySide6.QtCore`.
    root = (getattr(exc, "name", None) or "").split(".", 1)[0]
    if root in _QT_EXTRA_MODULES:
        return root
    # Import hooks and hand-raised ImportErrors may leave `.name` unset, so
    # fall back to the message text before giving up on the friendly path.
    text = str(exc)
    for module in sorted(_QT_EXTRA_MODULES):
        if module in text:
            return module
    return None


def run(argv: list[str] | None = None) -> int:
    """Launch the Qt GUI. Public entry point used by both `spacr-qt` and
    `python -m spacr.qt`.

    Args:
        argv: Optional CLI arguments. The first positional element,
              if present, opens directly into that app screen key
              (e.g. `spacr-qt mask`).

    Returns:
        The exit code returned by `QApplication.exec()`, or ``1`` when the
        optional Qt extra is not installed.
    """
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) == 1 and argv[0] in _VERSION_FLAGS:
        from spacr.version import get_version

        print(get_version())
        return 0

    try:
        from .app import launch
    except ImportError as exc:
        module = _missing_qt_extra(exc)
        if module is None:
            raise
        print(_QT_MISSING_MESSAGE.format(module=module), file=sys.stderr)
        return 1

    # Deliberately outside the `try`: an ImportError raised *during* a run —
    # a screen lazily importing an optional reader, say — is a real failure
    # and must not be reported as "Qt is not installed".
    return launch(argv)
