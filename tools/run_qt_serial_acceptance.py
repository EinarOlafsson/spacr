"""Keep the native serial Qt entry point importable by spawned workers."""
if __name__ == '__main__':
    import sys
    from PySide6.QtWidgets import QApplication

    app = QApplication([])
    if app.platformName() != 'xcb':
        raise SystemExit('N47 did not construct a real X-backed QApplication')
    app.setApplicationName('pytest-qt-qapp')

    import pytest
    raise SystemExit(pytest.main(sys.argv[1:]))
