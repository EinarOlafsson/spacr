"""Tutorial recordings run with Preferences -> Show alpha features off.

The maintainer's rule: nothing registered in ``spacr.settings.ALPHA_FEATURES``
gets a tutorial. Every capture path turns the preference off before the app
is built and refuses a frame while it is on; only a Preferences scene that
shows the toggle itself may opt in.
"""
import re
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QWidget

REPO = Path(__file__).resolve().parents[2]
TUTORIALS = REPO / "tools" / "tutorials"
sys.path.insert(0, str(TUTORIALS))
from capture_policy import (  # noqa: E402
    ALPHA_FEATURES_KEY,
    configure_appearance,
    force_alpha_features_off,
    force_alpha_features_off_in_profiles,
    verify_alpha_features_off,
    verify_appearance,
)


@pytest.fixture
def prefs(monkeypatch, tmp_path):
    from spacr.qt import preferences as prefs

    store = QSettings(str(tmp_path / "capture.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)
    monkeypatch.delenv("SPACR_NO_BACKDROP", raising=False)
    return prefs


@pytest.fixture
def dark_blobs_window(qtbot, prefs):
    from spacr.qt.widgets.ambient import AmbientWidget

    configure_appearance()
    window = QWidget()
    qtbot.addWidget(window)
    window.resize(400, 300)
    palette = window.palette()
    palette.setColor(QPalette.Window, QColor("#121212"))
    window.setPalette(palette)
    backdrop = AmbientWidget(window, theme="blobs")
    backdrop.setGeometry(window.rect())
    window.show()
    backdrop.show()
    qtbot.waitUntil(lambda: backdrop.frames_painted > 0)
    yield window
    backdrop.hide()


def test_the_key_is_the_one_preferences_stores():
    from spacr.qt import preferences

    assert ALPHA_FEATURES_KEY == preferences._KEY_SHOW_ALPHA_FEATURES


def test_configuring_a_recording_turns_alpha_features_off(prefs):
    prefs._set_show_alpha_features(True)
    configure_appearance()
    assert prefs._get_show_alpha_features() is False


def test_force_off_is_available_on_its_own(prefs):
    prefs._set_show_alpha_features(True)
    assert force_alpha_features_off() is True
    assert prefs._get_show_alpha_features() is False


def test_a_frame_is_refused_while_alpha_features_are_on(prefs, dark_blobs_window):
    assert verify_alpha_features_off() is False
    assert "alpha_features_shown_for_toggle_scene" not in verify_appearance(dark_blobs_window)
    prefs._set_show_alpha_features(True)
    with pytest.raises(RuntimeError, match="Capture refused: .*Show alpha features is on"):
        verify_alpha_features_off()
    with pytest.raises(RuntimeError, match="Show alpha features is on"):
        verify_appearance(dark_blobs_window)


def test_only_the_preferences_toggle_scene_may_opt_in(prefs, dark_blobs_window):
    prefs._set_show_alpha_features(True)
    assert verify_alpha_features_off(allow_alpha_toggle_scene=True) is True
    receipt = verify_appearance(dark_blobs_window, allow_alpha_toggle_scene=True)
    assert receipt["alpha_features_shown_for_toggle_scene"] is True


def test_profiles_prepared_before_launch_start_with_alpha_off(tmp_path):
    config_home = tmp_path / "config" / "mask"
    store_path = config_home / "spacr" / "qt.conf"
    store_path.parent.mkdir(parents=True)
    seeded = QSettings(str(store_path), QSettings.IniFormat)
    seeded.setValue(ALPHA_FEATURES_KEY, True)
    seeded.setValue("prefs/theme", "dark")
    seeded.sync()
    del seeded
    fresh = tmp_path / "profile" / ".config"

    written = force_alpha_features_off_in_profiles([config_home, fresh])

    assert written == [store_path, fresh / "spacr" / "qt.conf"]
    for path in written:
        store = QSettings(str(path), QSettings.IniFormat)
        assert str(store.value(ALPHA_FEATURES_KEY)).lower() == "false"
    kept = QSettings(str(store_path), QSettings.IniFormat)
    assert kept.value("prefs/theme") == "dark"


def test_the_neutral_wrapper_turns_alpha_off_before_it_launches_spacr():
    script = (TUTORIALS / "run_neutral_capture.sh").read_text()
    force = script.index("--force-alpha-off")
    assert force < script.index("exec ")
    assert '"$capture_stage/profile/.config"' in script
    assert '"$capture_stage"/config/*/' in script
    assert re.search(r'run_capped\.sh"? 2G .*\\\n\s*"\$capture_python" '
                     r'"\$capture_repo/tools/tutorials/capture_policy\.py"', script)


def test_capture_refresh_forces_off_before_the_window_and_passes_only_the_opt_in():
    source = (TUTORIALS / "capture_refresh.py").read_text()
    assert source.index("configure_appearance(args.theme, args.backdrop)") < \
        source.index("window = gui.MainWindow()")
    assert "'--preferences-alpha-toggle-scene', action='store_true'" in source
    calls = re.findall(r"verify_appearance\(([^)]*)\)", source)
    assert calls == ["\n            window, allow_alpha_toggle_scene=args.preferences_alpha_toggle_scene"]


def test_authoring_capture_sessions_refuse_frames_while_alpha_is_on(prefs):
    sys.path.insert(0, str(TUTORIALS / "authoring" / "tools"))
    import capture_mask_experiment

    capture_mask_experiment.refuse_alpha_features()
    prefs._set_show_alpha_features(True)
    with pytest.raises(RuntimeError, match="Show alpha features is on"):
        capture_mask_experiment.refuse_alpha_features()
    source = Path(capture_mask_experiment.__file__).read_text()
    save = source[source.index("    def save("):]
    assert save.index("refuse_alpha_features()") < save.index("self.window.grab()")
