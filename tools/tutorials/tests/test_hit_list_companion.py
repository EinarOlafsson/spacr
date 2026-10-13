"""The companion must remain visibly distinct and retain the real job owner."""
from pathlib import Path
import sys

from PySide6.QtWidgets import QApplication, QLabel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hit_list_companion import create_window, TITLE
from spacr.qt.screens.hit_list import HitListScreen


def test_real_widget_has_no_preloaded_or_substituted_results():
    app = QApplication.instance() or QApplication([])
    window, screen = create_window()
    window.show(); app.processEvents()
    try:
        assert type(screen) is HitListScreen
        assert screen.isVisible() and screen._browse_button.isEnabled()
        assert screen.hits() is None and screen.active_jobs() == 0
        assert window.windowTitle() == TITLE
        assert 'not the Regression shortcut' in TITLE
        disclosures = [w.text() for w in window.findChildren(QLabel)]
        assert any('EXTERNAL TUTORIAL COMPANION' in text and
                   'does not fix' in text for text in disclosures)
    finally:
        window.close(); app.processEvents()


def test_parent_close_calls_real_screen_shutdown(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window, screen = create_window()
    shutdown = screen._jobs.shutdown
    called = []
    def checked_shutdown(*args, **kwargs):
        called.append(True)
        return shutdown(*args, **kwargs)
    monkeypatch.setattr(screen._jobs, 'shutdown', checked_shutdown)
    window.show(); app.processEvents()
    assert not called
    window.close(); app.processEvents()
    assert called == [True]
    assert screen.active_jobs() == 0
