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
