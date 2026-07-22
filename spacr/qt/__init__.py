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

__all__ = ["run"]


def run(argv: list[str] | None = None) -> int:
    """Launch the Qt GUI. Public entry point used by both `spacr-qt` and
    `python -m spacr.qt`.

    Args:
        argv: Optional CLI arguments. The first positional element,
              if present, opens directly into that app screen key
              (e.g. `spacr-qt mask`).

    Returns:
        The exit code returned by `QApplication.exec()`.
    """
    from .app import launch
    return launch(argv)
