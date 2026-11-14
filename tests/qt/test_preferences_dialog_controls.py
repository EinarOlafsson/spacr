"""The built Preferences dialog: its dropdowns, Reset, Quit, the Fractal tab
and the live backdrop and icon switches a saved choice sets off.

Every test drives the real :class:`spacr.qt.preferences.PreferencesDialog`
(or the switch it calls) over a throwaway INI store and pins what the user is
left with: the row a dropdown opens on, the value Save writes, the words a
warning says, and which window closes when they quit.
"""
from __future__ import annotations

import os
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QSettings                             # noqa: E402
from PySide6.QtWidgets import (QApplication, QComboBox, QDialog,  # noqa: E402
                               QDialogButtonBox, QDoubleSpinBox, QFormLayout,
                               QLabel, QMessageBox, QPushButton, QSlider,
                               QSpinBox, QWidget)

from spacr.qt import preferences as prefs                        # noqa: E402
from spacr.qt import shutdown, theme                             # noqa: E402
from spacr.qt.widgets import ambient                             # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def store(qapp, tmp_path, monkeypatch):
    """A throwaway INI store, so no test touches the real preferences."""
    settings = QSettings(str(tmp_path / "spacr-qt.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: settings)
    monkeypatch.setattr(prefs, "_SAFE_MODE", False)
    assert str(tmp_path) in os.path.normpath(prefs._settings().fileName())
    return settings


@pytest.fixture
def spaceout(monkeypatch):
    """Spaceout on for one test (it is a process flag), off again after."""
    monkeypatch.setattr(theme, "_SPACEOUT", True)


def _dialog(qtbot):
    """A built Preferences dialog, registered for teardown."""
    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    return dialog


def _row(dialog, label_text):
    """The control on the row the user reads as ``label_text``."""
    for form in dialog.findChildren(QFormLayout):
        for index in range(form.rowCount()):
            label = form.itemAt(index, QFormLayout.LabelRole)
            widget = label.widget() if label is not None else None
            if isinstance(widget, QLabel) and widget.text() == label_text:
                return form.itemAt(index, QFormLayout.FieldRole).widget()
    raise AssertionError(f"no row labelled {label_text!r}")


def _save(dialog):
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()


#: Each dropdown that walks its own items looking for a stored value: the
#: getter it reads, the row the user sees it on, and a token no build offers.
_POINTED_AT_A_STORED_VALUE = (
    ("get_theme_choice", "Theme", "space:nebula"),
    ("get_ambient_animation", "Animation", "lava lamp"),
    ("get_color_blind_mode", "Colour-blind mode", "tetrachromacy"),
    ("get_figure_format", "Figure format", "bmp"),
    ("get_figure_png_dpi", "PNG resolution", 999),
    ("get_performance_level", "Performance", "hyperdrive"),
)

#: Dropdowns found by object name: language (its labels follow the stored
#: language) and the drift direction (built whatever the animation is).
_BY_NAME = (
    ("get_language", "LanguagePreference", "kl"),
    ("get_ambient_drift_direction", "AmbientDriftDirection", "sideways"),
)


def _combos(dialog):
    """The dropdowns under test, by the row label or object name they carry."""
    found = {getter: _row(dialog, label)
             for getter, label, _u in _POINTED_AT_A_STORED_VALUE}
    found.update({getter: dialog.findChild(QComboBox, name)
                  for getter, name, _u in _BY_NAME})
    return found


def _items(combo):
    return tuple(combo.itemData(i) for i in range(combo.count()))


def _open_with(qtbot, monkeypatch, answers, find):
    """Build the dialog while each getter answers as given; read it meanwhile."""
    with monkeypatch.context() as patch:
        for getter, value in answers.items():
            patch.setattr(prefs, getter, lambda value=value: value)
        dialog = _dialog(qtbot)
        return {getter: (combo.currentIndex(), combo.currentData(),
                         _items(combo))
                for getter, combo in find(dialog).items()}


def test_each_dropdown_opens_on_the_stored_value_or_its_first_row(
        store, qtbot, monkeypatch, subtests):
    """A stored token the dropdown holds is selected; an unknown one opens on row 0."""
    unknown = {getter: token for getter, _l, token in
               _POINTED_AT_A_STORED_VALUE + _BY_NAME}
    opened = _open_with(qtbot, monkeypatch, unknown, _combos)
    for getter, (index, data, items) in opened.items():
        with subtests.test(msg=f"unknown {getter}"):
            assert len(items) > 1 and unknown[getter] not in items
            assert (index, data) == (0, items[0])

    # The stored language also sets the language the row labels are drawn
    # in, so it is pointed at its last row in a dialog read by object name.
    last = {getter: items[-1] for getter, (_i, _d, items) in opened.items()}
    others = {g: v for g, v in last.items() if g != "get_language"}
    pointed = _open_with(qtbot, monkeypatch, others, _combos)
    pointed.update(_open_with(
        qtbot, monkeypatch, {"get_language": last["get_language"]},
        lambda dialog: {"get_language": dialog.findChild(
            QComboBox, "LanguagePreference")}))
    for getter, (index, data, items) in pointed.items():
        with subtests.test(msg=f"last {getter}"):
            assert items == opened[getter][2]
            assert data == last[getter] != items[0]
            assert index == len(items) - 1


def test_reset_moves_a_dropdown_only_to_a_default_it_holds(
        store, qtbot, monkeypatch, subtests):
    """Reset reads the real getters; an absent or foreign default leaves the row alone."""
    dialog = _dialog(qtbot)
    reset = dialog.findChild(QPushButton, "PreferencesReset")
    colour = _row(dialog, "Colour-blind mode")
    language = dialog.findChild(QComboBox, "LanguagePreference")
    colour.setCurrentIndex(colour.count() - 1)
    chosen = colour.currentData()
    assert chosen != colour.itemData(0)
    language_before = language.currentIndex()
    assert language_before >= 0
    cases = (
        ("foreign default", "get_color_blind_mode",
         lambda: "tetrachromacy", colour, colour.count() - 1),
        ("no language default", "get_language", lambda: None, language,
         language_before),
        ("held default", "get_color_blind_mode",
         lambda: colour.itemData(0), colour, 0),
    )
    for case, getter, answer, combo, expected_index in cases:
        with subtests.test(msg=case):
            with monkeypatch.context() as patch:
                patch.setattr(prefs, getter, answer)
                reset.click()
            assert combo.currentIndex() == expected_index


def test_the_button_box_offers_save_and_cancel_with_reset_apart(store, qtbot):
    """Save and Cancel carry this build's words; Reset sits in the reset role."""
    dialog = _dialog(qtbot)
    buttons = dialog.findChild(QDialogButtonBox)
    save = buttons.button(QDialogButtonBox.Save)
    cancel = buttons.button(QDialogButtonBox.Cancel)
    reset = dialog.findChild(QPushButton, "PreferencesReset")
    assert (save.text(), cancel.text()) == ("Save", "Cancel")
    assert [buttons.buttonRole(b) for b in (save, cancel, reset)] == [
        QDialogButtonBox.AcceptRole, QDialogButtonBox.RejectRole,
        QDialogButtonBox.ResetRole]


class _Handle:
    app_key = "segmentation"

    def elapsed(self):
        return 180.0


class _UnreadableHandle:
    """A handle with no module name whose elapsed time cannot be read."""

    def elapsed(self):
        raise RuntimeError("the clock went away")


class _Registry:
    def __init__(self, handle):
        self.cancelled = []
        self.running = [handle]

    def active(self):
        return list(self.running)

    def cancel_all(self, reason=""):
        self.cancelled.append(reason)
        self.running = []


def test_a_child_search_brings_waiting_pages_back_only_outside_the_shown_walk(
        store, qtbot, subtests):
    """An outside findChildren puts waiting pages back; the window's own walk leaves them away."""
    for case, window_only, walking, back in (
            ("an outside search", 0, False, True),
            ("during the window's own walk", 1, False, False),
            ("while the shown tree is walked", 0, True, False)):
        with subtests.test(msg=case):
            dialog = _dialog(qtbot)
            dialog._send_the_unseen_pages_away()
            away = dict(dialog._pages_away)
            assert len(away) == dialog._page_tabs.count() - 1
            dialog._window_only = window_only
            dialog.setProperty("spacrWalkingShownTree", walking)
            found = dialog.findChildren(QWidget)
            dialog._window_only = 0
            dialog.setProperty("spacrWalkingShownTree", False)
            assert dialog._pages_away == ({} if back else away)
            assert [page in found for page in away.values()] == [back] * len(away)


def test_the_quit_button_cancels_forces_or_leaves_gracefully(
        store, qtbot, monkeypatch, subtests):
    """Cancel keeps the window, Force leaves at once, graceful cancels and watches."""
    named = "Still running:\n  • segmentation — running for 3 min"
    unnamed = "Still running:\n  • job — running for 0 min"
    cases = (
        # (case, answer, handle, detail, forced, cancelled, watched, visible)
        ("cancel", shutdown.CANCEL, _Handle(), named, [], [], False, True),
        ("force", shutdown.FORCE, _Handle(), named, [True], [], False, True),
        ("graceful", shutdown.GRACEFUL, _Handle(), named, [],
         ["quit from Preferences"], True, False),
        ("graceful, unreadable handle", shutdown.GRACEFUL, _UnreadableHandle(),
         unnamed, [], ["quit from Preferences"], True, False),
        ("graceful without a registry", "", None, "", [], [], True, False),
    )
    real_watcher = shutdown.GracefulQuitWatcher
    for (case, answer, handle, detail, forced, cancelled, watched,
         visible) in cases:
        with subtests.test(msg=case):
            asked, forced_calls, watchers = [], [], []

            class _Watcher(real_watcher):
                """The real watcher, remembering how it was built and started."""

                def __init__(self, window, busy, **kwargs):
                    super().__init__(window, busy, **kwargs)
                    self.window, self.busy, self.kwargs = window, busy, kwargs

                def start(self):
                    super().start()
                    watchers.append(self)

            with monkeypatch.context() as patch:
                patch.setattr(shutdown, "ask_how_to_quit",
                              lambda *a, **k: asked.append(k) or answer)
                patch.setattr(shutdown, "force_quit_now",
                              lambda *a, **k: forced_calls.append(True))
                patch.setattr(shutdown, "GracefulQuitWatcher", _Watcher)
                dialog = _dialog(qtbot)
                registry = _Registry(handle) if handle is not None else None
                if registry is not None:
                    dialog._runs = registry
                dialog.show()
                dialog.findChild(QPushButton, "QuitSpacrButton").click()

            assert asked == [{"what": "spaCR", "detail": detail}]
            assert forced_calls == forced
            assert (registry.cancelled if registry else []) == cancelled
            assert dialog.isVisible() is visible
            assert len(watchers) == int(watched)
            if watched:
                watcher = watchers[0]
                assert watcher.window is dialog
                assert watcher.kwargs["what"] == "spaCR"
                assert watcher.busy() is False
                assert watcher.kwargs["describe"]() == ""
                # Nothing is still running, so the five-minute cycle never starts.
                assert watcher._timer.isActive() is False
                assert dialog.result() == QDialog.Accepted


# ---------------------------------------------------------------------------
# The Fractal tab (spaceout only)
# ---------------------------------------------------------------------------

def test_the_backend_note_names_the_renderer_this_machine_will_use(
        store, spaceout, qtbot, monkeypatch, subtests):
    """Each reason Automatic cannot use the GPU gets its own sentence."""
    from spacr.qt.widgets import fractal_travel

    dialog = _dialog(qtbot)
    backend = dialog.findChild(QComboBox, "FractalBackend")
    note = dialog.findChild(QLabel, "FractalBackendNote")

    def choose(key):
        backend.setCurrentIndex(backend.findData(key))
        return note.text()

    cases = (
        ("no display", False, True, "gpu",
         "No usable display/OpenGL context"),
        ("no vispy", True, False, "auto", "VisPy is not installed"),
        ("gpu usable", True, True, "auto",
         "Automatic: this machine will use the"),
    )
    for case, opengl, vispy, key, fragment in cases:
        with subtests.test(msg=case):
            monkeypatch.setattr(fractal_travel, "platform_can_do_opengl",
                                lambda opengl=opengl: opengl)
            monkeypatch.setattr(fractal_travel, "gpu_is_available",
                                lambda vispy=vispy: vispy)
            choose("cpu")
            text = choose(key)
            assert fragment in text
    with subtests.test(msg="cpu note differs from automatic"):
        cpu_text = choose("cpu")
        assert "renderer" in cpu_text and "Automatic" not in cpu_text
        assert choose("auto").startswith("Automatic: this machine will use the")


def test_saving_the_fractal_tab_writes_clamps_and_explains(
        store, spaceout, qtbot, monkeypatch, caplog, subtests):
    """Save stores every field, clamps an unusable number with a warning, and
    survives a backdrop that cannot restart."""
    from spacr.qt.widgets import fractal_travel

    warned = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(
        lambda *args, **kwargs: warned.append(args[2])))
    with subtests.test(msg="fields are stored"):
        dialog = _dialog(qtbot)
        dialog.findChild(QComboBox, "FractalQuality").setCurrentIndex(
            dialog.findChild(QComboBox, "FractalQuality").findData("ultra"))
        dialog.findChild(QSpinBox, "FractalSupersampling").setValue(3)
        dialog.findChild(QDoubleSpinBox, "FractalScale").setValue(1.5)
        _save(dialog)
        saved = prefs.get_fractal_settings()
        assert (saved["quality"], saved["supersampling"]) == ("ultra", 3)
        assert saved["scale"] == pytest.approx(1.5)
        assert warned == []
    for value, expected, warning in ((0, 1, True), (2, 2, False)):
        with subtests.test(msg=f"supersampling {value}"):
            warned.clear()
            dialog = _dialog(qtbot)
            dialog.findChild(QSpinBox, "FractalSupersampling").setValue(value)
            _save(dialog)
            assert prefs.get_fractal_settings()["supersampling"] == expected
            if warning:
                assert len(warned) == 1
                assert ("supersampling: 0 is too small (needs at least 1)"
                        in warned[0])
            else:
                assert warned == []
    with subtests.test(msg="backdrop cannot restart"):
        def gone():
            raise RuntimeError("the backdrop widget has been destroyed")

        monkeypatch.setattr(fractal_travel, "apply_saved_controls", gone)
        dialog = _dialog(qtbot)
        dialog.findChild(QSpinBox, "FractalSupersampling").setValue(4)
        with caplog.at_level("DEBUG", logger=prefs.LOG.name):
            _save(dialog)
        assert "could not restart the dive" in caplog.text
        assert prefs.get_fractal_settings()["supersampling"] == 4


# ---------------------------------------------------------------------------
# Density, the live spaceout switch and fold icons
# ---------------------------------------------------------------------------

def _theme_combo(dialog):
    return dialog.findChild(QComboBox, "AmbientTheme") or next(
        box for box in dialog.findChildren(QComboBox)
        if box.findData("data_art_impulse_lens") >= 0
        and box.findData("blobs") >= 0)


def test_the_density_slider_never_follows_a_theme_change(
        store, qtbot, qt_theme_applied, monkeypatch, subtests):
    """Neither an unsaved default nor a saved density moves when the theme does."""
    monkeypatch.setattr(ambient.AmbientWidget, "_start_producer",
                        lambda self: None)
    cases = (
        # (case, animation, stored density, start, user move, saved)
        ("default then moved", "data_art_impulse_lens", None, 10, 50, 0.5),
        ("saved density", "blobs", 0.1, 10, None, None),
    )
    for case, animation, stored, start, moved, saved in cases:
        with subtests.test(msg=case):
            store.clear()
            prefs.set_ambient_animation(animation)
            if stored is None:
                assert not store.contains(prefs._KEY_AMBIENT_DENSITY)
                assert prefs.get_ambient_density() == pytest.approx(0.1)
            else:
                store.setValue(prefs._KEY_AMBIENT_DENSITY, stored)
            dialog = _dialog(qtbot)
            density = dialog.findChild(QSlider, "AmbientDensity")
            combo = _theme_combo(dialog)
            assert density.value() == start
            if moved is not None:
                density.setValue(moved)
            held = density.value()
            for other in ("blobs", "data_art_impulse_lens", "blobs"):
                combo.setCurrentIndex(combo.findData(other))
                assert density.value() == held, other
            if saved is not None:
                _save(dialog)
                assert float(store.value(prefs._KEY_AMBIENT_DENSITY)) == \
                    pytest.approx(saved)


class _Fractal(QWidget):
    """A spaceout backdrop stand-in that records pause and retirement."""

    def __init__(self, parent):
        super().__init__(parent)
        self._spaceout_built_from = ("test",)
        self._paused = False
        self.retired = False

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def is_paused(self):
        return self._paused

    def shutdown(self):
        self.retired = True


def _host(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(320, 200)
    host.show()
    return host


def _switch():
    ambient._apply_spaceout_animation_choice(QApplication.instance())


@pytest.fixture
def spaceout_store(tmp_path, monkeypatch):
    """A private store with spaceout enabled for the process, restored after."""
    settings = QSettings(str(tmp_path / "preferences.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: settings)
    settings.setValue("prefs/brand_palette_revision", True)
    was = theme.spaceout_enabled()
    theme.enable_spaceout()
    yield settings
    theme.enable_spaceout() if was else theme.disable_spaceout()


def test_the_spaceout_animation_switch_keeps_each_backdrops_state(
        qtbot, spaceout_store, monkeypatch, subtests):
    """Off pauses and remembers, on restores, a theme swap retires the old one."""
    with subtests.test(msg="off and on restore each fractal"):
        host = _host(qtbot)
        running, hidden_paused = _Fractal(host), _Fractal(host)
        running.show()
        hidden_paused.pause()
        hidden_paused.hide()
        host._ambient = running

        prefs.set_ambient_animation("none")
        assert prefs.get_ambient_enabled() is False
        _switch()
        assert running.is_paused() and running.isHidden()
        assert [running.property(p) for p in (
            "spacrPreferenceDisabled", "spacrPreferenceWasHidden")] == [
            True, False]
        assert [hidden_paused.property(p) for p in (
            "spacrPreferenceWasHidden", "spacrPreferenceWasPaused")] == [
            True, True]
        _switch()
        assert running.property("spacrPreferenceWasHidden") is False

        prefs.set_ambient_animation(ambient.SPACEOUT_THEME)
        assert prefs.get_ambient_theme() == ambient.SPACEOUT_THEME
        _switch()
        assert not running.isHidden() and not running.is_paused()
        assert running.property("spacrPreferenceDisabled") is False
        assert hidden_paused.isHidden() and hidden_paused.is_paused()
        assert hidden_paused.property("spacrPreferenceDisabled") is False
        assert host._ambient is running and running.retired is False
        _switch()
        assert host._ambient is running and not running.is_paused()

        theme.disable_spaceout()
        prefs.set_ambient_animation("none")
        _switch()
        assert running.isHidden() and running.is_paused()
        assert [running.property(p) for p in (
            "spacrPreferenceDisabled", "spacrPreferenceWasHidden",
            "spacrPreferenceWasPaused")] == [True, False, False]
        theme.enable_spaceout()
    with subtests.test(msg="a disabled fractal replaced by a regular theme"):
        host = _host(qtbot)
        old = _Fractal(host)
        old.show()
        host._ambient = old
        prefs.set_ambient_animation("none")
        _switch()
        assert old.isHidden() and old.is_paused()
        prefs.set_ambient_animation("blobs")
        _switch()
        new = host._ambient
        assert isinstance(new, ambient.AmbientWidget)
        assert new.theme() == "blobs" and old.retired is True
        assert not new.isHidden() and new.is_animating()
    with subtests.test(msg="regular backdrop retires only for spaceout"):
        # Before any spaceout choice is stored, a spaceout launch dresses
        # every requested backdrop as the default evolving field.
        spaceout_store.remove(prefs._KEY_SPACEOUT_ANIMATION)
        assert ambient.dressed("blobs", "spacr") == (
            ambient.DEFAULT_SPACEOUT_THEME, ambient.DEFAULT_SPACEOUT_PALETTE)
        host = _host(qtbot)
        old = ambient.install_ambient(host, theme="blobs", palette="spacr")
        host._ambient = old
        old.hide()
        prefs.set_ambient_animation("drift")
        _switch()
        assert host._ambient is old

        def install(target, *, theme, palette):
            widget = _Fractal(target)
            widget.show()
            return widget

        with monkeypatch.context() as patch:
            patch.setattr(ambient, "install_ambient", install)
            prefs.set_ambient_animation(ambient.SPACEOUT_THEME)
            _switch()
        new = host._ambient
        assert isinstance(new, _Fractal) and new.isHidden()
        assert old.property("spacrRetiringBackdrop") is True
        assert old.parentWidget() is None


def test_a_theme_change_reinks_fold_icons_without_resizing_them(
        qtbot, qapp, subtests):
    """Fold buttons repaint in the new theme's icon and keep size and state."""
    from spacr.qt import app as app_module
    from spacr.qt.widgets.fold_strip import FoldButton

    for before, after in (("dark", "light"), ("light", "dark")):
        with subtests.test(msg=f"{before} to {after}"):
            owner = QWidget()
            qtbot.addWidget(owner)
            owner.refresh_theme = types.MethodType(
                app_module.MainWindow.refresh_theme, owner)
            owner.refresh_language = lambda: None
            prefs.set_theme(before)
            prefs.apply_preferences_to_app(qapp)
            button = FoldButton("train_cellpose", parent=owner, checkable=True)
            button.setChecked(True)
            geometry = (button.size(), button.iconSize())
            original = button.icon().pixmap(button.iconSize()).toImage()

            prefs.set_theme(after)
            prefs.apply_preferences_to_app(qapp)
            prefs._refresh_owner_window(owner)
            expected = app_module._icon_for_app("train_cellpose").pixmap(
                button.iconSize()).toImage()
            assert original != expected
            assert button.icon().pixmap(button.iconSize()).toImage() == expected
            assert (button.size(), button.iconSize()) == geometry
            assert button.isChecked() and button.app_key == "train_cellpose"
