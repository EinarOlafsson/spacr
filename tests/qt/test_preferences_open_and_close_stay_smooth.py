"""Opening and closing Preferences does not stall the GUI thread.

The tutorial capture gate found it: the second in which Preferences opened
showed 21 distinct frames and the second in which it closed 22, against 29
everywhere else, while the backdrop otherwise painted 56-58 fps. Measured
offscreen on a loaded machine, the open held the GUI thread for 0.46-1.5 s
(build 0.3-0.9 s, then the first show and paint) and the close for
0.9-1.4 s (Home rebuilt and the window retranslated after every close,
Cancel included); a Save with nothing changed held it for 3.7-4.9 s.

What replaced it, and what is tested here:

* the dialog is built and shown off screen while the user is idle, and the
  menu shows that one; it is kept after a close for as long as it shows
  what a fresh one would, and is otherwise dropped and rebuilt at idle;
* the window redraws after the close only when something it draws from
  changed;
* Apply, Cancel and Save mean what they meant, and a kept dialog shows the
  stored values every time it opens.

Timing is measured against a reference taken in the same test on the same
machine -- the old open and the old close -- so a loaded CI runner slows
both sides alike.
"""
from __future__ import annotations

import statistics
import time
import weakref

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QComboBox, QDialog, QDialogButtonBox, QMainWindow, QSlider,
)

from spacr.qt import preferences  # noqa: E402


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A private preference store, and warming allowed."""
    path = tmp_path / "preferences.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    monkeypatch.setenv("SPACR_NETWORK_CONFIG", str(tmp_path / "network.json"))
    monkeypatch.setenv("SPACR_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SPACR_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.delenv("SPACR_BENCHMARK_JSON", raising=False)
    monkeypatch.setattr(preferences, "_SAFE_MODE", False)
    preferences.set_language("en")
    preferences.set_theme_choice("dark")
    preferences.set_ambient_animation("none")
    preferences._settings().sync()
    yield path
    from spacr import updater

    updater._undo_network_exports()


def _stop_warming(state):
    """Stop the pending warming slice, if it has not already run."""
    try:
        state["timer"].stop()
    except RuntimeError:
        pass


class _Owner(QMainWindow):
    """A window that counts its redraws."""

    def __init__(self):
        super().__init__()
        self.refreshes = 0

    def refresh_theme(self):
        self.refreshes += 1


@pytest.fixture
def owner(qtbot, store):
    """A shown owner window whose Preferences are kept."""
    window = _Owner()
    qtbot.addWidget(window)
    window.resize(900, 700)
    window.show()
    qtbot.waitExposed(window)
    state = preferences._warm_the_preferences(window, 60000)
    assert state is not None
    yield window
    _stop_warming(state)
    for dialog in [state["dialog"], state["returned"], *state["retired"]]:
        if dialog is not None and preferences._widget_is_alive(dialog):
            dialog.hide()
            dialog.deleteLater()


def _warm(window, qtbot):
    """Run warming slices until a dialog is built and shown off screen."""
    for _ in range(40):
        state = preferences._preferences_warmth(window)
        dialog = state["dialog"]
        if (dialog is not None
                and getattr(dialog, "_spacr_shown_off_screen", False)
                and state["returned"] is None and not state["retired"]):
            return dialog
        preferences._warm_one_slice(window)
        qtbot.wait(1)
    raise AssertionError("Preferences was never warmed")


def _settle(window, qtbot):
    """Let the slice that looks at a closed dialog run."""
    preferences._warm_one_slice(window)
    qtbot.wait(1)


def _theme_combo(dialog):
    return next(combo for combo in dialog.findChildren(QComboBox)
                if combo.findData("glass") >= 0)


def _until_painted(dialog, qtbot):
    """Deliver the show's layout and first paint, and nothing that waits."""
    QApplication.processEvents()
    QApplication.processEvents()


def _close(dialog, which):
    dialog.findChild(QDialogButtonBox).button(which).click()


def _cache_failure_details(window, references=()):
    """Describe live cache decisions only after an owning assertion fails.

    :param window: the Preferences owner.
    :param references: weak references observed outside measured intervals.
    :returns: identities and predicate differences, never stored values.
    """
    from collections import Counter

    state = preferences._preferences_warmth(window) or {}
    candidates = [state.get("dialog"), state.get("returned"),
                  *state.get("retired", ()), *(ref() for ref in references)]
    details = {
        "cached_identity": id(state["dialog"]) if state.get("dialog") is not None else None,
        "returned_identity": id(state["returned"]) if state.get("returned") is not None else None,
        "observed_identities": [id(ref()) if ref() is not None else None
                                for ref in references],
        "owner_visible": window.isVisible(),
        "mouse_buttons": str(QApplication.mouseButtons()),
        "modal_identity": id(QApplication.activeModalWidget())
                          if QApplication.activeModalWidget() is not None else None,
        "popup_identity": id(QApplication.activePopupWidget())
                          if QApplication.activePopupWidget() is not None else None,
        "dialogs": [],
    }
    seen = set()
    for dialog in candidates:
        if dialog is None or id(dialog) in seen:
            continue
        seen.add(id(dialog))
        row = {"identity": id(dialog), "native_alive": preferences._widget_is_alive(dialog)}
        details["dialogs"].append(row)
        if not row["native_alive"]:
            continue
        try:
            shown = getattr(dialog, "_spacr_shows", None)
            row.update(reusable=getattr(dialog, "_reusable", False),
                       snapshot_present=shown is not None,
                       confirmation_pending=getattr(dialog, "_apply_confirmation", None) is not None)
            if shown is None:
                continue
            store, world, draft = shown
            current_store = preferences._preferences_store_values()
            current_world = preferences._preferences_world()
            current_draft = preferences._preferences_draft(dialog)
            row["changed_store_keys"] = sorted(
                key for key in store.keys() | current_store.keys()
                if store.get(key) != current_store.get(key))
            row["changed_world_indices"] = [
                index for index in range(max(len(world), len(current_world)))
                if world[index:index + 1] != current_world[index:index + 1]]
            row["draft_equal"] = draft == current_draft
            row["draft_control_counts"] = (len(draft[0]), len(current_draft[0]))
            row["draft_order_only"] = (draft[0] != current_draft[0]
                and Counter(map(repr, draft[0])) == Counter(map(repr, current_draft[0])))
            row["draft_extra_equal"] = draft[1] == current_draft[1]
            row["changed_controls"] = sorted({
                entry[:2] for entry in draft[0]
                if entry not in current_draft[0]} | {
                entry[:2] for entry in current_draft[0]
                if entry not in draft[0]})[:12]
        except Exception as exc:
            row["diagnostic_error_type"] = type(exc).__name__
    return details


def test_cache_failure_details_report_changed_controls_and_deleted_wrappers(
        owner, qtbot):
    """The diagnostic reads real controls and tolerates a retained dead wrapper."""
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QCheckBox

    dialog = QDialog(owner)
    dialog._reusable = True
    checkbox = QCheckBox(dialog)
    checkbox.setObjectName("CacheDiagnosticControl")
    state = preferences._preferences_warmth(owner)
    state["dialog"] = dialog
    reference = weakref.ref(dialog)
    preferences._note_what_the_dialog_shows(dialog)
    checkbox.setChecked(True)
    details = _cache_failure_details(owner, (reference,))
    row = details["dialogs"][0]
    assert row["native_alive"]
    assert not row["draft_equal"]
    assert row["changed_controls"] == [("CacheDiagnosticControl", "QCheckBox")]
    dialog.deleteLater()
    QCoreApplication.sendPostedEvents(dialog, QEvent.DeferredDelete)
    details = _cache_failure_details(owner, (reference,))
    assert details["dialogs"][0] == {"identity": id(dialog), "native_alive": False}
    state["dialog"] = None


# --------------------------------------------------------------------------
# the open and the close stay inside a frame of the old ones
# --------------------------------------------------------------------------

def test_the_open_is_a_small_fraction_of_building_the_dialog(owner, qtbot):
    """A warm open costs a fraction of the build-and-show it replaced.

    The reference is the old open, measured here: build, show, first paint.
    """
    old = []
    for _ in range(2):
        started = time.perf_counter()
        fresh = preferences.PreferencesDialog(owner)
        fresh.show()
        _until_painted(fresh, qtbot)
        old.append(time.perf_counter() - started)
        fresh.hide()
        fresh.deleteLater()
        qtbot.wait(1)

    new = []
    observations = []
    for _ in range(3):
        observations.append(weakref.ref(_warm(owner, qtbot)))
        _stop_warming(preferences._preferences_warmth(owner))
        started = time.perf_counter()
        dialog = preferences._a_preferences_for(owner)
        dialog.show()
        _until_painted(dialog, qtbot)
        new.append(time.perf_counter() - started)
        observations.append(weakref.ref(dialog))
        _close(dialog, QDialogButtonBox.Cancel)
        preferences._put_the_preferences_away(owner, dialog)
        _settle(owner, qtbot)

    reference = min(old)
    worst = statistics.median(new)
    assert worst < 0.25 * reference, (
        f"opening a warm Preferences took {worst * 1000:.0f} ms against "
        f"{reference * 1000:.0f} ms for building one; the open is meant "
        f"to be a show of a dialog that is already built; "
        f"cache={_cache_failure_details(owner, observations)}")


def test_the_close_is_a_small_fraction_of_the_old_redraw(
        qtbot, qt_theme_applied, store, monkeypatch):
    """Cancel no longer rebuilds Home, retranslates the window or leaks.

    The reference is what every close used to do, ``refresh_theme``,
    measured on the same real window.
    """
    from PySide6.QtCore import QTimer

    from spacr.qt.app import MainWindow

    from PySide6.QtCore import QEventLoop

    def _exec(dialog):
        """``exec`` as the menu runs it, answered by the timer below."""
        loop = QEventLoop()
        dialog.finished.connect(loop.quit)
        dialog.show()
        loop.exec()
        return dialog.result()

    monkeypatch.setattr(preferences._preferences_window_class(), "exec",
                        _exec)
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.waitExposed(window)
    preferences._warm_the_preferences(window, 60000)
    try:
        started = time.perf_counter()
        window.refresh_theme()
        reference = time.perf_counter() - started

        redraws = []
        real_refresh = window.refresh_theme
        monkeypatch.setattr(window, "refresh_theme",
                            lambda: redraws.append(1) or real_refresh())
        closes = []
        observations = []
        for _ in range(2):
            observations.append(weakref.ref(_warm(window, qtbot)))
            _stop_warming(preferences._preferences_warmth(window))

            def _cancel():
                dialog = next(
                    w for w in QApplication.topLevelWidgets()
                    if isinstance(w, preferences._preferences_window_class())
                    and w.isVisible())
                closes.append(time.perf_counter())
                _close(dialog, QDialogButtonBox.Cancel)

            QTimer.singleShot(200, _cancel)
            window.show_preferences_on()
            closes[-1] = time.perf_counter() - closes[-1]
            _settle(window, qtbot)

        assert redraws == [], "a Cancel that changed nothing redrew the window"
        worst = max(closes)
        assert worst < 0.25 * reference, (
            f"closing Preferences took {worst * 1000:.0f} ms against "
            f"{reference * 1000:.0f} ms for the redraw it used to do; "
            f"cache={_cache_failure_details(window, observations)}")
        state = preferences._preferences_warmth(window)
        assert state["dialog"] is not None, (
            "the closed dialog was not kept; "
            f"cache={_cache_failure_details(window, observations)}")
        kept = [w for w in window.findChildren(QDialog)
                if isinstance(w, preferences._preferences_window_class())]
        assert len(kept) == 1, f"{len(kept)} Preferences windows outlived their close"
    finally:
        _stop_warming(preferences._preferences_warmth(window))


def test_a_close_that_changed_the_look_still_redraws(owner, qtbot):
    """The redraw is skipped only when it has nothing to do."""
    dialog = _warm(owner, qtbot)
    dialog = preferences._a_preferences_for(owner)
    dialog.show()
    combo = _theme_combo(dialog)
    combo.setCurrentIndex(combo.findData("light"))
    _close(dialog, QDialogButtonBox.Save)
    assert owner.refreshes == 1
    assert not preferences._the_window_needs_refreshing(dialog)
    preferences._put_the_preferences_away(owner, dialog)


# --------------------------------------------------------------------------
# the kept dialog means what a fresh one meant
# --------------------------------------------------------------------------

def test_a_clean_cancel_keeps_the_dialog_and_it_shows_the_store(owner, qtbot):
    first = _warm(owner, qtbot)
    dialog = preferences._a_preferences_for(owner)
    assert dialog is first
    dialog.show()
    _close(dialog, QDialogButtonBox.Cancel)
    preferences._put_the_preferences_away(owner, dialog)
    _settle(owner, qtbot)

    again = preferences._a_preferences_for(owner)
    assert again is first, "a dialog that changed nothing was rebuilt"
    assert _theme_combo(again).currentData() == "dark"
    again.reject()


def test_a_cancelled_edit_is_not_shown_again(owner, qtbot):
    _warm(owner, qtbot)
    dialog = preferences._a_preferences_for(owner)
    dialog.show()
    combo = _theme_combo(dialog)
    combo.setCurrentIndex(combo.findData("light"))
    dialog.findChild(QSlider, "PaneOpacity").setValue(37)
    _close(dialog, QDialogButtonBox.Cancel)
    preferences._put_the_preferences_away(owner, dialog)
    _settle(owner, qtbot)

    assert preferences.get_theme_choice() == "dark"
    again = preferences._a_preferences_for(owner)
    assert again is not dialog, "a dialog holding a cancelled edit was kept"
    assert _theme_combo(again).currentData() == "dark"
    assert again.findChild(QSlider, "PaneOpacity").value() != 37
    again.deleteLater()


def test_a_store_changed_elsewhere_is_shown_on_the_next_open(owner, qtbot):
    kept = _warm(owner, qtbot)
    preferences.set_theme_choice("light")
    dialog = preferences._a_preferences_for(owner)
    assert dialog is not kept, "a dialog built before the change was shown"
    assert _theme_combo(dialog).currentData() == "light"
    dialog.deleteLater()


def test_save_writes_and_the_next_open_shows_what_was_saved(owner, qtbot):
    _warm(owner, qtbot)
    dialog = preferences._a_preferences_for(owner)
    dialog.show()
    combo = _theme_combo(dialog)
    combo.setCurrentIndex(combo.findData("light"))
    _close(dialog, QDialogButtonBox.Save)
    assert not dialog.isVisible()
    assert preferences.get_theme_choice() == "light"
    preferences._put_the_preferences_away(owner, dialog)
    _settle(owner, qtbot)

    again = preferences._a_preferences_for(owner)
    assert _theme_combo(again).currentData() == "light"
    again.deleteLater()


def test_a_save_that_changed_nothing_keeps_the_dialog(owner, qtbot):
    first = _warm(owner, qtbot)
    dialog = preferences._a_preferences_for(owner)
    dialog.show()
    _close(dialog, QDialogButtonBox.Save)
    assert owner.refreshes == 0, "a Save that changed nothing redrew the window"
    preferences._put_the_preferences_away(owner, dialog)
    _settle(owner, qtbot)
    assert preferences._a_preferences_for(owner) is first


def test_apply_keep_then_cancel_is_kept_and_reopens_with_it(owner, qtbot):
    from PySide6.QtWidgets import QMessageBox

    _warm(owner, qtbot)
    dialog = preferences._a_preferences_for(owner)
    dialog.show()
    combo = _theme_combo(dialog)
    combo.setCurrentIndex(combo.findData("light"))
    _close(dialog, QDialogButtonBox.Apply)
    qtbot.waitUntil(lambda: dialog._apply_confirmation is not None)
    question = dialog._apply_confirmation
    assert isinstance(question, QMessageBox)
    next(b for b in question.buttons() if b.text() == "Keep").click()
    qtbot.waitUntil(lambda: dialog._apply_confirmation is None)
    assert dialog.isVisible(), "Apply closed Preferences"
    _close(dialog, QDialogButtonBox.Cancel)
    assert preferences.get_theme_choice() == "light", "Cancel undid a kept Apply"
    preferences._put_the_preferences_away(owner, dialog)
    _settle(owner, qtbot)

    again = preferences._a_preferences_for(owner)
    assert _theme_combo(again).currentData() == "light"
    again.deleteLater()


def test_apply_revert_then_cancel_shows_the_reverted_value(owner, qtbot):
    _warm(owner, qtbot)
    dialog = preferences._a_preferences_for(owner)
    dialog.show()
    combo = _theme_combo(dialog)
    combo.setCurrentIndex(combo.findData("light"))
    _close(dialog, QDialogButtonBox.Apply)
    qtbot.waitUntil(lambda: dialog._apply_confirmation is not None)
    question = dialog._apply_confirmation
    next(b for b in question.buttons() if b.text() == "Revert").click()
    qtbot.waitUntil(lambda: dialog._apply_confirmation is None)
    _close(dialog, QDialogButtonBox.Cancel)
    assert preferences.get_theme_choice() == "dark"
    preferences._put_the_preferences_away(owner, dialog)
    _settle(owner, qtbot)

    again = preferences._a_preferences_for(owner)
    assert again is not dialog, "the draft left after a Revert was kept"
    assert _theme_combo(again).currentData() == "dark"
    again.deleteLater()


def test_a_dialog_is_let_go_when_preferences_are_not_kept(qtbot, store):
    """Without warming each closed dialog is deleted instead of leaking."""
    window = _Owner()
    qtbot.addWidget(window)
    dialog = preferences._a_preferences_for(window)
    dialog.show()
    dialog.reject()
    assert preferences._widget_is_alive(dialog)
    preferences._put_the_preferences_away(window, dialog)
    qtbot.waitUntil(lambda: not preferences._widget_is_alive(dialog))
    assert not preferences._widget_is_alive(dialog)
    fresh = preferences._a_preferences_for(window)
    assert fresh is not dialog
    fresh.deleteLater()


# --------------------------------------------------------------------------
# the kept dialog is compared on everything a build reads
# --------------------------------------------------------------------------

def test_every_key_a_build_reads_is_in_the_compared_groups(
        qtbot, tmp_path, monkeypatch):
    """A key read from a group outside the list would go stale unseen."""
    path = tmp_path / "recorded.ini"
    read = set()

    class _Recording(QSettings):
        def value(self, key, *args, **kwargs):
            group = self.group()
            read.add(f"{group}/{key}" if group else key)
            return super().value(key, *args, **kwargs)

        def contains(self, key):
            group = self.group()
            read.add(f"{group}/{key}" if group else key)
            return super().contains(key)

    monkeypatch.setattr(preferences, "_settings",
                        lambda: _Recording(str(path), QSettings.IniFormat))
    monkeypatch.setattr(preferences, "_SAFE_MODE", False)
    preferences._set_show_alpha_features(True)
    read.clear()
    dialog = preferences.PreferencesDialog(None)
    qtbot.addWidget(dialog)
    outside = sorted(key for key in read
                     if key.split("/", 1)[0]
                     not in preferences._PREFERENCES_STORE_GROUPS)
    assert read, "nothing was recorded; the store was not the one read"
    assert outside == [], (
        f"Preferences reads {outside}, outside _PREFERENCES_STORE_GROUPS; "
        f"a kept dialog would not notice them change")


def test_page_detachment_does_not_change_the_same_dialogs_draft(qtbot, store):
    """Reparenting preserves control identity while edits remain significant."""
    from PySide6.QtWidgets import QCheckBox, QLineEdit, QScrollArea, QWidget
    from shiboken6 import getCppPointer

    dialog = QDialog()
    qtbot.addWidget(dialog)
    dialog._reusable = True
    dialog._pages_away = {}
    pages, holders, checks, texts = [], [], [], []
    for index in range(2):
        holder = QScrollArea(dialog)
        page = QWidget()
        check = QCheckBox(page)
        check.setObjectName("SharedPreferenceName")
        check.setChecked(bool(index))
        text = QLineEdit("[1, 2]", page)
        text.setObjectName("OrderedPreferenceText")
        holder.setWidget(page)
        holders.append(holder)
        pages.append(page)
        checks.append(check)
        texts.append(text)
    ordered_extra = ["first", "second"]
    dialog._draft_extras = lambda: ordered_extra
    native_controls = [getCppPointer(control)[0] for control in checks + texts]
    preferences._note_what_the_dialog_shows(dialog)
    assert preferences._still_shows_the_store(dialog, saved_by_it=True)

    for index, holder in enumerate(holders):
        page = holder.takeWidget()
        assert page is pages[index]
        dialog._pages_away[index] = page
    assert preferences._still_shows_the_store(dialog, saved_by_it=True)
    dialog._pages_away = dict(reversed(tuple(dialog._pages_away.items())))
    assert preferences._still_shows_the_store(dialog, saved_by_it=True)
    for index, holder in enumerate(holders):
        holder.setWidget(dialog._pages_away.pop(index))
    assert [getCppPointer(control)[0] for control in checks + texts] == native_controls
    assert preferences._still_shows_the_store(dialog, saved_by_it=True)

    checks[0].setChecked(True)
    checks[1].setChecked(False)
    assert not preferences._still_shows_the_store(dialog, saved_by_it=True)
    checks[0].setChecked(False)
    checks[1].setChecked(True)
    assert preferences._still_shows_the_store(dialog, saved_by_it=True)
    texts[0].setText("[2, 1]")
    assert not preferences._still_shows_the_store(dialog, saved_by_it=True)
    texts[0].setText("[1, 2]")
    ordered_extra.reverse()
    assert not preferences._still_shows_the_store(dialog, saved_by_it=True)
