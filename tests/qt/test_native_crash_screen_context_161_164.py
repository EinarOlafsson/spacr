"""GitHub #161, #164: the main window records the shown screen for a native crash report."""
from types import SimpleNamespace

from spacr import crashreport
from spacr.qt import app


def _window(current, screens, startup=None):
    return SimpleNamespace(_stack=SimpleNamespace(currentWidget=lambda: current), _screens=screens,
                           _startup=startup)


def test_the_shown_screen_is_recorded_by_its_english_title(monkeypatch):
    noted = []
    monkeypatch.setattr(crashreport, "_note_native_context", lambda **fields: noted.append(fields))
    monkeypatch.setattr(app, "APPS", [("mask", "Generate Masks", "", "")])
    screen, home = object(), object()
    app._note_native_screen(_window(screen, {"mask": screen}, home))
    app._note_native_screen(_window(home, {"mask": screen}, home))
    app._note_native_screen(_window(object(), {"mask": screen}, home))
    assert noted == [{"screen": "Generate Masks"}, {"screen": "Home"}]


def test_closing_the_window_records_each_teardown_step_170(qtbot, qt_theme_applied, monkeypatch):
    """GitHub #170: a segfault inside closeEvent named no step; each step of
    the close is now recorded for the next start's report."""
    noted = []
    monkeypatch.setattr(crashreport, "_note_native_context", lambda **fields: noted.append(fields))
    monkeypatch.setattr(app, "_end_the_event_loop", lambda _app: None)
    window = app.MainWindow()
    qtbot.addWidget(window)
    window.show()
    noted.clear()
    window.close()
    assert [fields["action"] for fields in noted if "action" in fields] == [
        "close stops running jobs", "close closes the screens", "close ends the session",
        "close shuts the consoles", "close releases update workers", "close closes the window",
        "close ends the event loop"]
