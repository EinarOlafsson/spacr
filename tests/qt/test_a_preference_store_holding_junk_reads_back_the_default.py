"""A preference the store cannot parse reads back as its default, and every
helper that pushes a preference into the running app survives a part of the
app that will not take it.

The preference store is a text file the user can edit and an older build
can have written. What these tests pin is what a user gets back from it
when a value in it is junk: the default, never an exception at startup.
The second half breaks one collaborator at a time around the helpers that
push preferences out -- the tooltip policy, the colour-scheme reader, the
sound, the icon re-sizer, the disk report's worker -- and asserts the
preference is still stored and the button or stylesheet the user looks at
is still right.
"""
from __future__ import annotations

import os
import types

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings, QSize  # noqa: E402
from PySide6.QtWidgets import QPushButton, QToolButton, QWidget  # noqa: E402

from spacr.qt import preferences as prefs  # noqa: E402

pytestmark = pytest.mark.qt


def _boom(*_args, **_kwargs):
    raise RuntimeError("this part will not answer")


@pytest.fixture
def store():
    """The sandboxed store, with every key a test writes put back after."""
    settings = prefs._settings()
    before = {}

    def put(key, value):
        if key not in before:
            before[key] = (settings.contains(key), settings.value(key))
        settings.setValue(key, value)
        settings.sync()

    def remember(key):
        if key not in before:
            before[key] = (settings.contains(key), settings.value(key))

    put.remember = remember
    yield put
    settings = prefs._settings()
    for key, (had, value) in before.items():
        if had:
            settings.setValue(key, value)
        else:
            settings.remove(key)
    settings.sync()


# ---------------------------------------------------------------------------
# Junk in the store
# ---------------------------------------------------------------------------

def test_every_scale_and_width_reads_junk_as_its_default(store):
    store(prefs._KEY_RUNTIME_TEXT_SCALE, "large please")
    store(prefs._KEY_GUI_SCALE, "huge")
    store(prefs._KEY_DOCK_WIDTH, "wide")

    assert prefs.get_runtime_text_scale() == prefs.DEFAULT_RUNTIME_TEXT_SCALE
    assert prefs.get_gui_scale() == prefs.DEFAULT_GUI_SCALE
    assert prefs.get_dock_width() == 0


def test_a_dock_width_that_is_not_a_number_is_stored_as_the_fitting_width(
        store):
    store.remember(prefs._KEY_DOCK_WIDTH)

    prefs.set_dock_width("wide")

    assert prefs.get_dock_width() == 0


def test_an_opacity_that_is_not_a_number_is_stored_as_the_default(store):
    store.remember(prefs._KEY_PANE_OPACITY)

    prefs.set_pane_opacity("opaque")

    assert prefs.get_pane_opacity() == pytest.approx(
        prefs.DEFAULT_PANE_OPACITY_PCT / 100.0)


def test_a_layout_decision_that_is_not_a_record_is_no_decision(store):
    store(prefs._KEY_LAYOUT_DECISION, "[1, 2, 3]")
    assert prefs._get_layout_decision() == {}

    store(prefs._KEY_LAYOUT_DECISION, "{not json")
    assert prefs._get_layout_decision() == {}


def test_a_layout_decision_that_json_cannot_write_is_not_stored(store):
    store(prefs._KEY_LAYOUT_DECISION, "")

    prefs._set_layout_decision({"width": object()})

    assert prefs._get_layout_decision() == {}


def test_an_unknown_dock_mode_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown dock mode 'floating'"):
        prefs.set_dock_mode("floating")


def test_a_news_height_is_remembered_and_junk_is_not(store):
    store.remember(prefs._KEY_NEWS_HEIGHT)

    assert prefs.set_news_height(-40) == 0
    assert prefs.set_news_height(260) == 260
    assert prefs.get_news_height() == 260
    assert prefs.set_news_height("tall") == 0
    assert prefs.get_news_height() == 260

    store(prefs._KEY_NEWS_HEIGHT, "tall")
    assert prefs.get_news_height() == 0


def test_a_dashboard_panel_remembers_when_it_was_cleared(store):
    for key in prefs.DASHBOARD_WATERMARKS.values():
        store.remember(key)

    assert prefs.set_dashboard_watermark("no such panel") == ""
    assert prefs.get_dashboard_watermark("no such panel") == ""
    prefs.clear_dashboard_watermark("no such panel")

    stamped = prefs.set_dashboard_watermark("runs")
    assert stamped and "T" in stamped
    assert prefs.get_dashboard_watermark("runs") == stamped
    assert prefs.set_dashboard_watermark(
        "totals", "2026-01-02T03:04:05+00:00") == "2026-01-02T03:04:05+00:00"

    prefs.clear_dashboard_watermark("runs")
    assert prefs.get_dashboard_watermark("runs") == ""
    assert prefs.get_dashboard_watermark("totals") == (
        "2026-01-02T03:04:05+00:00")


def test_a_volume_that_is_not_a_number_plays_at_the_default(store):
    store(prefs._KEY_SOUND_VOLUME, "loud")
    assert prefs.get_sound_volume() == prefs.DEFAULT_SOUND_VOLUME

    store(prefs._KEY_SOUND_VOLUME, "nan")
    assert prefs.get_sound_volume() == prefs.DEFAULT_SOUND_VOLUME

    assert prefs.set_sound_volume("loud") == prefs.DEFAULT_SOUND_VOLUME
    assert prefs.get_sound_volume() == prefs.DEFAULT_SOUND_VOLUME


def test_an_unknown_sound_set_or_event_is_refused(store):
    store.remember(prefs._KEY_SOUND_THEME)

    with pytest.raises(ValueError, match="unknown sound set"):
        prefs.set_sound_theme("no such set")
    with pytest.raises(KeyError):
        prefs.set_sound_event_enabled("no such event", True)


def test_the_music_bed_plays_when_the_level_cannot_be_read(monkeypatch):
    monkeypatch.setattr(prefs, "get_performance_level", _boom)

    assert prefs.sound_bed_rests() is False


def test_sound_is_not_offered_when_spaceout_cannot_say(monkeypatch):
    from spacr.qt import theme

    monkeypatch.setattr(theme, "spaceout_enabled", _boom)

    assert prefs.sound_is_offered() is False


# ---------------------------------------------------------------------------
# Pushing a preference out, when a part of the app will not take it
# ---------------------------------------------------------------------------

def test_a_system_theme_whose_scheme_cannot_be_read_is_dark(monkeypatch):
    from spacr.qt import theme

    monkeypatch.setattr(prefs, "get_theme", lambda: "system")
    monkeypatch.setattr(theme, "system_colour_scheme", _boom)

    assert prefs.resolve_effective_theme() == "dark"


def test_the_tooltip_switch_is_stored_when_the_policy_cannot_refresh(
        store, monkeypatch):
    from spacr.qt import tooltip_policy

    store.remember(prefs._KEY_TOOLTIPS_ENABLED)
    monkeypatch.setattr(tooltip_policy, "invalidate_tooltip_policy", _boom)

    prefs.set_tooltips_enabled(False)
    assert prefs.get_tooltips_enabled() is False
    prefs.set_tooltips_enabled(True)
    assert prefs.get_tooltips_enabled() is True


class _Store(QSettings):
    """A real store whose key lookup fails half way through."""

    def contains(self, _key):
        raise RuntimeError("the store file went away")


def test_a_store_that_fails_while_the_space_keys_are_removed_is_left(
        tmp_path):
    store = _Store(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat)
    store.setValue("prefs/theme", "dark")

    prefs._forget_the_space_theme_keys(store)

    assert store.value("prefs/theme") == "dark"
    assert str(store.fileName()) not in prefs._SPACE_KEYS_CLEARED


def test_icons_are_resized_past_a_widget_that_refuses(qapp, qtbot,
                                                      monkeypatch):
    monkeypatch.setattr(prefs, "get_font_scale", lambda: 2.0)
    refuses = QWidget()
    qtbot.addWidget(refuses)
    setattr(refuses, prefs._ICON_SCALE_HOOK, _boom)
    button = QToolButton()
    qtbot.addWidget(button)
    button.setProperty(prefs._KEY_ICON_BASE_W, 16)
    button.setIconSize(QSize(16, 16))

    moved = prefs._rescale_icon_sizes(qapp)

    assert moved >= 1
    assert button.iconSize().width() > 16


def test_with_no_application_no_icon_is_resized(monkeypatch):
    from PySide6.QtWidgets import QApplication

    monkeypatch.setattr(QApplication, "instance", staticmethod(lambda: None))

    assert prefs._rescale_icon_sizes() == 0


# ---------------------------------------------------------------------------
# The disk report's button
# ---------------------------------------------------------------------------

class _Parent:
    """A dialog stand-in whose visibility is whatever the test says."""

    def __init__(self, visible):
        self._visible = visible

    def isVisible(self):
        if isinstance(self._visible, BaseException):
            raise self._visible
        return self._visible


def test_a_report_is_answered_only_while_someone_is_asking():
    assert prefs._still_asking(None) is True
    assert prefs._still_asking(_Parent(True)) is True
    assert prefs._still_asking(_Parent(RuntimeError("gone"))) is False
    assert prefs._still_asking(_Parent(AttributeError("odd"))) is True
    assert prefs._disk_button(None) is None
    assert prefs._disk_button(object()) is None


def test_only_a_live_widget_is_safe_to_touch(qapp, monkeypatch):
    import shiboken6

    widget = QWidget()
    assert prefs._widget_is_alive(None) is False
    assert prefs._widget_is_alive(widget) is True

    monkeypatch.setattr(shiboken6, "isValid", _boom)
    assert prefs._widget_is_alive(widget) is True
    assert prefs._widget_is_alive(
        types.SimpleNamespace(objectName=lambda: (_ for _ in ()).throw(
            RuntimeError("deleted")))) is False
    widget.deleteLater()


class _SyncRunner:
    """Runs the job on the spot, as the worker would, or refuses it."""

    def __init__(self, accept=True):
        self.accept = accept

    def submit(self, job, done):
        if not self.accept:
            return False
        done(job())
        return True


def _disk_dialog(qtbot):
    dialog = QWidget()
    qtbot.addWidget(dialog)
    button = QPushButton("Check disk space", dialog)
    button.setObjectName("CheckDiskButton")
    button.setToolTip("How full is the disk?")
    dialog.show()
    return dialog, button


def test_a_disk_read_that_fails_gives_the_button_back(qapp, qtbot,
                                                      monkeypatch):
    from spacr.qt import resource_cleanup

    shown = []
    monkeypatch.setattr(resource_cleanup, "disk_report", _boom)
    monkeypatch.setattr(prefs, "_disk_report_runner", lambda: _SyncRunner())
    monkeypatch.setattr(prefs, "_show_resource_result",
                        lambda *a: shown.append(a))
    dialog, button = _disk_dialog(qtbot)

    prefs._start_disk_report(dialog)

    assert button.isEnabled()
    assert button.toolTip() == "How full is the disk?"
    assert shown == []


def test_a_disk_report_after_the_dialog_closed_is_not_shown(qapp, qtbot,
                                                            monkeypatch):
    from spacr.qt import resource_cleanup

    shown = []
    monkeypatch.setattr(resource_cleanup, "disk_report", lambda: "report")
    monkeypatch.setattr(prefs, "_disk_report_runner", lambda: _SyncRunner())
    monkeypatch.setattr(prefs, "_show_resource_result",
                        lambda *a: shown.append(a))
    dialog, button = _disk_dialog(qtbot)
    dialog.hide()

    prefs._start_disk_report(dialog)
    assert shown == [] and button.isEnabled()

    dialog.show()
    monkeypatch.setattr(prefs, "_show_resource_result", _boom)
    prefs._start_disk_report(dialog)
    assert button.isEnabled()


def test_a_disk_read_the_worker_refuses_gives_the_button_back(qapp, qtbot,
                                                              monkeypatch):
    monkeypatch.setattr(prefs, "_disk_report_runner",
                        lambda: _SyncRunner(accept=False))
    dialog, button = _disk_dialog(qtbot)

    prefs._start_disk_report(dialog)

    assert button.isEnabled()
    assert button.toolTip() == "How full is the disk?"


def test_a_disk_runner_built_for_the_other_threading_is_retired(
        qapp, monkeypatch):
    class _OldRunner:
        def cancel(self):
            raise RuntimeError("already deleted")

    old = _OldRunner()
    retired = []
    monkeypatch.setattr(prefs, "_DISK_RUNNER", old)
    monkeypatch.setattr(prefs, "_DISK_RUNNER_THREADED", False)
    monkeypatch.setattr(prefs, "_RETIRED_DISK_RUNNERS", retired)

    runner = prefs._disk_report_runner()

    assert runner is not old
    assert retired == [old]
    assert prefs._disk_report_runner() is runner


# ---------------------------------------------------------------------------
# Applying everything, when the optional parts fail
# ---------------------------------------------------------------------------

def test_preferences_still_apply_when_the_optional_parts_fail(
        qapp, monkeypatch):
    import sys

    from spacr.qt import tooltip_policy
    from spacr.qt.widgets import preview_scale

    monkeypatch.setattr(tooltip_policy, "install_tooltip_policy", _boom)
    monkeypatch.setitem(sys.modules, "spacr.qt.widgets.preview_scale",
                        preview_scale)
    monkeypatch.setattr(preview_scale, "refresh_all_preview_scales", _boom)
    monkeypatch.setattr(prefs, "get_sound_enabled", lambda: True)
    import spacr.qt.sound as sound

    monkeypatch.setattr(sound, "apply_sound_preferences", _boom)
    monkeypatch.setattr(prefs, "_rescale_icon_sizes", _boom)
    monkeypatch.setattr(qapp, "_spacr_preferences_style_signature", None,
                        raising=False)

    prefs.apply_preferences_to_app(qapp)

    assert qapp.property("spacrLanguage") == prefs.get_language()
    assert qapp._spacr_preferences_stylesheet
