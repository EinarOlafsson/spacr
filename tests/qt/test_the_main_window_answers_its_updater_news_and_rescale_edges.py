"""The main window's update, news, rescale and dock helpers at their edges.

Each helper here is driven on a stand-in window (a ``SimpleNamespace`` with
just the parts the helper reads) because the edge in question -- a worker
whose C++ side is already gone, a Home page torn down before the news came
back, a restart record that cannot be saved, a window that is closing --
cannot be arranged on a real ``MainWindow`` without contorting it. What is
asserted is what the user is then shown (the status bar, a message box's
text, where focus lands, the hint strip) or what is kept (the scale a
screen is recorded at, the values a rebuilt form is handed).
"""
from __future__ import annotations

import os
import types

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from spacr.qt import app as app_mod  # noqa: E402
from spacr.qt import preferences as prefs_mod  # noqa: E402

pytestmark = pytest.mark.qt

MainWindow = app_mod.MainWindow


def _boom(*_args, **_kwargs):
    raise RuntimeError("this part is gone")


class _StatusBar:
    def __init__(self):
        self.messages = []

    def showMessage(self, text, _ms=0):
        self.messages.append(text)


class _Boxes:
    """QMessageBox's static entry points, recorded instead of shown."""

    def __init__(self, monkeypatch):
        self.warnings = []
        self.informations = []
        fake = types.SimpleNamespace(
            warning=lambda _p, title, text: self.warnings.append(
                (title, text)),
            information=lambda _p, title, text: self.informations.append(
                (title, text)),
            Yes=object())
        monkeypatch.setattr(app_mod, "QMessageBox", fake)


class _DeadWorker:
    """A worker whose C++ object was deleted under the Python wrapper."""

    def isRunning(self):
        raise RuntimeError("Internal C++ object already deleted.")


class _BusyWorker:
    def isRunning(self):
        return True


def _window(**parts):
    bar = _StatusBar()
    window = types.SimpleNamespace(statusBar=lambda: bar, _closing=False,
                                   **parts)
    return window, bar


# ---------------------------------------------------------------------------
# Check for updates
# ---------------------------------------------------------------------------

def test_a_second_update_check_while_one_runs_says_so_and_starts_nothing():
    started = []
    window, bar = _window(_update_worker=_BusyWorker(),
                          _start_update_worker=lambda *a: started.append(a))

    MainWindow._check_for_updates(window)

    assert bar.messages == ["An update operation is already running."]
    assert started == []


def test_a_worker_that_is_already_deleted_does_not_block_a_new_check():
    started = []
    window, bar = _window(_update_worker=_DeadWorker(),
                          _on_update_check_done=object(),
                          _start_update_worker=lambda *a: started.append(a))

    MainWindow._check_for_updates(window)

    assert bar.messages == ["Checking for updates…"]
    assert [call[0] for call in started] == ["check"]


# ---------------------------------------------------------------------------
# The News refresh
# ---------------------------------------------------------------------------

def test_a_news_refresh_already_under_way_is_not_doubled(monkeypatch):
    monkeypatch.setattr(prefs_mod, "get_refresh_news", lambda: True)
    busy = _BusyWorker()
    window, _bar = _window(_news_worker=busy)

    MainWindow._refresh_news(window)

    assert window._news_worker is busy


def test_a_news_refresh_with_no_reader_installed_starts_no_worker(
        monkeypatch):
    import spacr.updater as updater

    monkeypatch.setattr(prefs_mod, "get_refresh_news", lambda: True)
    monkeypatch.delattr(updater, "fetch_release_notes")
    dead = _DeadWorker()
    window, bar = _window(_news_worker=dead)

    MainWindow._refresh_news(window)

    assert window._news_worker is dead
    assert bar.messages == []


def test_news_that_arrives_after_home_is_gone_is_dropped_quietly():
    applied = []
    window, _bar = _window(_startup=None)
    MainWindow._on_news_ready(window, ["v9"])

    page = types.SimpleNamespace(apply_release_news=_boom)
    window._startup = page
    MainWindow._on_news_ready(window, ["v9"])

    window._startup = types.SimpleNamespace(
        apply_release_news=applied.append)
    MainWindow._on_news_ready(window, ["v9"])
    assert applied == [["v9"]]


def test_a_failed_news_refresh_opens_no_message_box(monkeypatch, caplog):
    boxes = _Boxes(monkeypatch)
    window, bar = _window()

    with caplog.at_level("DEBUG", logger=app_mod.LOG.name):
        MainWindow._on_news_failed(window, "news", "Traceback: offline")

    assert boxes.warnings == [] and bar.messages == []
    assert "Traceback: offline" in caplog.text


# ---------------------------------------------------------------------------
# Upgrades
# ---------------------------------------------------------------------------

def test_an_upgrade_that_failed_for_want_of_pip_names_the_command(
        monkeypatch):
    import spacr.updater as updater

    boxes = _Boxes(monkeypatch)
    monkeypatch.setattr(updater, "find_uv", lambda: "/opt/spacr/uv")
    window, _bar = _window()

    MainWindow._on_upgrade_done(
        window, (1, "Collecting...\n/usr/bin/python: No module named pip\n"))

    (title, text), = boxes.warnings
    assert title == "Updates"
    assert "pip returned exit code 1" in text
    assert "This environment has no pip" in text
    assert "/opt/spacr/uv pip install --upgrade" in text


def test_an_upgrade_whose_restart_cannot_be_saved_keeps_the_window_open(
        monkeypatch):
    from spacr import restart_state

    boxes = _Boxes(monkeypatch)
    monkeypatch.setattr(restart_state, "save", _boom)
    closed = []
    screen = types.SimpleNamespace(
        app_key="measure",
        _settings_model=types.SimpleNamespace(collect=lambda: {"a": 1}))
    window, _bar = _window(
        _stack=types.SimpleNamespace(currentWidget=lambda: screen),
        close=lambda: closed.append(1))

    MainWindow._restart_after_package_upgrade(window)

    (title, text), = boxes.warnings
    assert "could not be saved for restart" in text
    assert closed == []
    assert not getattr(window, "_restart_after_update", False)


def test_old_install_and_update_results_during_shutdown_show_nothing(
        monkeypatch):
    boxes = _Boxes(monkeypatch)
    window, bar = _window()
    window._closing = True

    MainWindow._on_old_installs_found(window, [object()])
    MainWindow._on_update_sequence_done(window, ([], None))
    MainWindow._on_upgrade_done(window, (1, "boom"))

    assert boxes.warnings == [] and boxes.informations == []
    assert bar.messages == []


# ---------------------------------------------------------------------------
# Rescaling a cached screen
# ---------------------------------------------------------------------------

def test_a_screen_never_recorded_at_a_scale_is_not_stale():
    window = types.SimpleNamespace(_screen_scales={})

    assert MainWindow._screen_scale_is_stale(window, "measure") is False


def test_a_screen_built_at_another_scale_is_stale(monkeypatch):
    monkeypatch.setattr(app_mod, "_current_font_scale", lambda: 1.25)
    window = types.SimpleNamespace(_screen_scales={"a": 1.0, "b": 1.25})

    assert MainWindow._screen_scale_is_stale(window, "a") is True
    assert MainWindow._screen_scale_is_stale(window, "b") is False


def _rescale_window(screens):
    rebuilt = []
    window = types.SimpleNamespace(
        _screens=screens, _screen_scales={},
        rebuild_app_screen=lambda key, values: rebuilt.append((key, values)))
    return window, rebuilt


def test_a_form_is_rebuilt_at_the_new_scale_with_its_values(monkeypatch):
    model = types.SimpleNamespace(collect=lambda: {"channels": [1, 2]})
    window, rebuilt = _rescale_window(
        {"measure": types.SimpleNamespace(_settings_model=model)})

    MainWindow._rebuild_for_scale(window, "measure")
    MainWindow._rebuild_for_scale(window, "not open")

    assert rebuilt == [("measure", {"channels": [1, 2]})]


def test_a_screen_with_no_form_or_an_unreadable_one_is_only_rerecorded(
        monkeypatch):
    monkeypatch.setattr(app_mod, "_current_font_scale", lambda: 1.5)
    broken = types.SimpleNamespace(
        _settings_model=types.SimpleNamespace(collect=_boom))
    window, rebuilt = _rescale_window(
        {"annotate": types.SimpleNamespace(), "measure": broken})

    MainWindow._rebuild_for_scale(window, "annotate")
    MainWindow._rebuild_for_scale(window, "measure")

    assert rebuilt == []
    assert window._screen_scales == {"annotate": 1.5, "measure": 1.5}


# ---------------------------------------------------------------------------
# The dock and its hints
# ---------------------------------------------------------------------------

def test_an_unreadable_dock_preference_reads_as_locked(monkeypatch):
    monkeypatch.setattr(prefs_mod, "get_dock_mode", _boom)

    assert MainWindow.dock_mode(types.SimpleNamespace()) == "locked"


def _hint_window(handler):
    page = types.SimpleNamespace(show_module_hint=handler)
    return types.SimpleNamespace(
        _stack=types.SimpleNamespace(currentWidget=lambda: page))


def test_a_hover_hint_is_written_even_when_its_summary_cannot_be_read(
        monkeypatch):
    import spacr.qt.i18n_module_summaries as summaries

    shown = []
    window = _hint_window(lambda key, text: shown.append((key, text)))
    monkeypatch.setattr(summaries, "module_summary", _boom)
    key = app_mod.APPS[0][0]

    MainWindow._show_module_hint(window, "")
    MainWindow._show_module_hint(window, key)

    assert shown == [(key, "")]


def test_a_hint_strip_that_refuses_the_text_does_not_break_the_hover():
    window = _hint_window(_boom)

    assert MainWindow._show_module_hint(window, app_mod.APPS[0][0]) is None


class _Row:
    def __init__(self):
        self.focused = []

    def setFocus(self, reason):
        self.focused.append(reason)


def test_the_dock_shortcut_focuses_the_first_row_only_when_it_can():
    first, second = _Row(), _Row()
    window = types.SimpleNamespace(
        _dock_mode="locked",
        _sidebar=types.SimpleNamespace(rows=lambda: [first, second]))
    MainWindow.toggle_app_drawer(window)
    assert len(first.focused) == 1 and second.focused == []

    window._sidebar = types.SimpleNamespace(rows=_boom)
    MainWindow.toggle_app_drawer(window)
    window._sidebar = types.SimpleNamespace(rows=lambda: [])
    MainWindow.toggle_app_drawer(window)
    assert len(first.focused) == 1


# ---------------------------------------------------------------------------
# Preferences on a tab
# ---------------------------------------------------------------------------

class _FakeDialog:
    opened = []

    def __init__(self, parent):
        self.parent = parent

    def exec(self):
        _FakeDialog.opened.append(self)
        return 0


def test_preferences_open_even_when_the_tab_cannot_be_found(monkeypatch):
    import spacr.qt.preferences_navigation as nav

    _FakeDialog.opened = []
    monkeypatch.setattr(prefs_mod, "PreferencesDialog", _FakeDialog)
    monkeypatch.setattr(nav, "show_tab", _boom)
    themed = []
    window = types.SimpleNamespace(refresh_theme=lambda: themed.append(1))

    found = MainWindow.show_preferences_on(window, "PreferencesTabTheme",
                                           "Theme")

    assert found is False
    assert len(_FakeDialog.opened) == 1
    assert themed == [1]


def test_preferences_open_on_the_named_tab(monkeypatch):
    import spacr.qt.preferences_navigation as nav

    _FakeDialog.opened = []
    asked = []
    monkeypatch.setattr(prefs_mod, "PreferencesDialog", _FakeDialog)
    monkeypatch.setattr(nav, "show_tab",
                        lambda dialog, tab, label: asked.append((tab, label))
                        or True)
    window = types.SimpleNamespace(refresh_theme=lambda: None)

    assert MainWindow.show_preferences_on(window, "PreferencesTabFonts") is True
    assert asked == [("PreferencesTabFonts", "")]
