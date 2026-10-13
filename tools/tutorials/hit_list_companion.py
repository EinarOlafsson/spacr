"""Explicit standalone tutorial launcher for spaCR's existing Hit List widget.

Run ``python hit_list_companion.py`` in the installed spaCR environment, then
Browse to the example's results folder. This is NOT the Regression shortcut,
does not repair that shortcut, and does not replace any spaCR implementation.
Only the existing widget reads, filters and exports the selected results.
"""
import sys


TITLE = 'Tutorial companion — Hit List (not the Regression shortcut)'


def create_window():
    """Wrap the unmodified, threaded Hit List screen with a visible disclosure."""
    from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget
    from spacr.qt.screens.hit_list import HitListScreen

    class CompanionWindow(QWidget):
        def closeEvent(self, event):
            # Parent close does not promise to call each child's closeEvent.
            # Use the real screen's worker-draining close implementation.
            screen.close()
            super().closeEvent(event)

    window = CompanionWindow()
    window.setWindowTitle(TITLE)
    layout = QVBoxLayout(window)
    disclosure = QLabel(
        'EXTERNAL TUTORIAL COMPANION — the existing spaCR Hit List widget.\n'
        'This does not fix Regression’s hidden Hits panel. Browse to results; '
        'no analysis is rerun. Investigate integration is not demonstrated here.')
    disclosure.setWordWrap(True)
    layout.addWidget(disclosure)
    screen = HitListScreen(parent=window)
    layout.addWidget(screen, 1)
    window.resize(1600, 1000)
    return window, screen


def main():
    """Launch a genuine screen in its own process using normal app preferences."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from spacr.qt.preferences import apply_preferences_to_app

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs)
    app = QApplication(sys.argv)
    apply_preferences_to_app(app)
    window, screen = create_window()
    window.show()
    result = app.exec()
    # Keep the screen and its worker owner alive for the entire event loop.
    if screen.active_jobs():
        raise RuntimeError('The companion closed while a Hit List job was active')
    return result


if __name__ == '__main__':
    raise SystemExit(main())
