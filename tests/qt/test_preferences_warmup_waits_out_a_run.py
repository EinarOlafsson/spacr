"""The kept Preferences window is not rebuilt in the middle of a run.

Lesson 08, clip batch_20_batch_settings, stalled for 0.39 s about 9 s after
Run while the warm-up glassed a Preferences window it had just rebuilt. No
preference had changed. The first figure of a run on a new profile calls
``get_figure_colors``, whose one-time migration writes the marker
``prefs/figure_color_scale``; the key landed in a group the kept window
compares, so at its next check the window was out of date and was built
again. The What's New check writes ``updates/last_seen_version`` the same
way.

Two things are tested here: bookkeeping keys no row shows do not make the
window out of date, and while a run is registered the warm-up builds,
drops and re-sheets nothing, and catches up once the run ends. A real
preference change still rebuilds it.

Every check is calibrated in the same test: the bookkeeping write is shown
to have made the old comparison fail, and the warm-up slices during the run
are compared with the build measured on the same machine.
"""
from __future__ import annotations

import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QComboBox, QMainWindow, QSlider  # noqa: E402

from spacr.qt import bridge, preferences  # noqa: E402


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A private preference store with no migration markers yet."""
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


@pytest.fixture
def owner(qtbot, store):
    """A shown window whose Preferences are kept, warming by hand."""
    window = QMainWindow()
    qtbot.addWidget(window)
    window.resize(900, 700)
    window.show()
    qtbot.waitExposed(window)
    state = preferences._warm_the_preferences(window, 60000)
    assert state is not None
    yield window
    try:
        state["timer"].stop()
    except RuntimeError:
        pass
    for dialog in [state["dialog"], state["returned"], *state["retired"]]:
        if dialog is not None and preferences._widget_is_alive(dialog):
            dialog.hide()
            dialog.deleteLater()


@pytest.fixture
def builds(monkeypatch):
    """Every Preferences build, as the time it took in seconds."""
    taken = []
    real = preferences.PreferencesDialog

    def _timed(parent=None):
        start = time.perf_counter()
        try:
            return real(parent)
        finally:
            taken.append(time.perf_counter() - start)

    monkeypatch.setattr(preferences, "PreferencesDialog", _timed)
    return taken


@pytest.fixture
def run():
    """Register a Measure run the way ``make_thread`` does, and drop it."""
    made = []

    def _start():
        def job(_settings):
            return None

        setattr(job, bridge.APP_KEY_ATTR, "measure")
        worker = bridge.PipelineWorker(job, {})
        handle = bridge.RunHandle("measure", worker, None,
                                  parent=bridge.registry())
        bridge.registry().register(handle)
        made.append(handle)
        return handle

    yield _start
    for handle in made:
        bridge.registry().unregister(handle)


def _slice(window, qtbot) -> float:
    """One warm-up slice and the events it posts; its length in seconds."""
    start = time.perf_counter()
    preferences._warm_one_slice(window)
    qtbot.wait(1)
    return time.perf_counter() - start


def _warm(window, qtbot):
    """Slice until a dialog is built and shown off screen."""
    for _ in range(40):
        state = preferences._preferences_warmth(window)
        dialog = state["dialog"]
        if (dialog is not None
                and getattr(dialog, "_spacr_shown_off_screen", False)
                and state["returned"] is None and not state["retired"]):
            return dialog
        _slice(window, qtbot)
    raise AssertionError("Preferences was never warmed")


def _raw_keys() -> set:
    store = preferences._settings()
    return set(getattr(store, "_real", store).allKeys())


def test_a_run_that_writes_only_bookkeeping_keeps_the_window(
        owner, qtbot, builds, run, monkeypatch):
    kept = _warm(owner, qtbot)
    assert len(builds) == 1
    build_s = builds[0]
    before = _raw_keys()
    assert preferences._KEY_FIG_COLOR_SCALE not in before

    handle = run()
    assert preferences._a_run_is_going()
    preferences.get_figure_colors()
    preferences._note_running_version("99.0.0")
    written = _raw_keys() - before
    assert preferences._KEY_FIG_COLOR_SCALE in written
    assert preferences._KEY_SEEN_VERSION in written

    with monkeypatch.context() as old:
        old.setattr(preferences, "_PREFERENCES_STORE_IGNORED",
                    frozenset({preferences._KEY_PREFERENCES_HELP_HEIGHT}))
        assert not preferences._still_shows_the_store(kept), (
            "calibration: the bookkeeping write no longer made the old "
            "comparison fail, so this test proves nothing")
    assert preferences._still_shows_the_store(kept)

    during = [_slice(owner, qtbot) for _ in range(6)]
    state = preferences._preferences_warmth(owner)
    assert len(builds) == 1, "the kept Preferences was rebuilt during a run"
    assert state["dialog"] is kept and state["retired"] == []
    assert max(during) < build_s / 4, (
        f"a warm-up slice during the run took {max(during) * 1000:.1f} ms "
        f"against a {build_s * 1000:.1f} ms build")

    bridge.registry().unregister(handle)
    assert not preferences._a_run_is_going()
    for _ in range(4):
        _slice(owner, qtbot)
    assert len(builds) == 1, "bookkeeping keys made the window out of date"
    assert state["dialog"] is kept
    assert preferences._a_preferences_for(owner) is kept


def test_a_preference_changed_during_a_run_rebuilds_after_it(
        owner, qtbot, builds, run):
    kept = _warm(owner, qtbot)
    assert len(builds) == 1

    handle = run()
    preferences.set_theme_choice("light")
    preferences.set_pane_opacity(0.55)
    assert not preferences._still_shows_the_store(kept)
    for _ in range(4):
        _slice(owner, qtbot)
    state = preferences._preferences_warmth(owner)
    assert len(builds) == 1, "the warm-up built Preferences during a run"
    assert state["dialog"] is kept and state["retired"] == []

    bridge.registry().unregister(handle)
    _slice(owner, qtbot)
    assert state["dialog"] is None and state["retired"] == [kept]
    fresh = _warm(owner, qtbot)
    assert len(builds) == 2, "a real preference change was not rebuilt"
    assert fresh is not kept
    combo = next(combo for combo in fresh.findChildren(QComboBox)
                 if combo.findData("glass") >= 0)
    assert combo.currentData() == "light"
    assert fresh.findChild(QSlider, "PaneOpacity").value() == 55
    assert preferences._a_preferences_for(owner) is fresh


def test_a_preference_changed_with_no_run_rebuilds_at_the_next_check(
        owner, qtbot, builds):
    kept = _warm(owner, qtbot)
    preferences.set_pane_opacity(0.6)
    _slice(owner, qtbot)
    state = preferences._preferences_warmth(owner)
    assert state["dialog"] is None and state["retired"] == [kept]
    fresh = _warm(owner, qtbot)
    assert len(builds) == 2 and fresh is not kept


def test_an_open_during_a_run_never_shows_an_out_of_date_window(
        owner, qtbot, builds, run):
    kept = _warm(owner, qtbot)
    run()
    preferences.set_pane_opacity(0.45)
    _slice(owner, qtbot)
    dialog = preferences._a_preferences_for(owner)
    assert dialog is not kept
    assert dialog.findChild(QSlider, "PaneOpacity").value() == 45
    dialog.deleteLater()
