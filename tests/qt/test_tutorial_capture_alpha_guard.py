"""Tutorial recordings run with Preferences -> Show alpha features off.

The maintainer's rule: nothing registered in ``spacr.settings.ALPHA_FEATURES``
gets a tutorial. Every capture path turns the preference off before the app
is built and refuses a frame while it is on; only a Preferences scene that
shows the toggle itself may opt in.
"""
import os
import shutil
import subprocess
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


@pytest.mark.parametrize("host_override", [False, True])
@pytest.mark.parametrize("policy_status", [0, 23])
@pytest.mark.skipif(sys.platform != "linux", reason="The bwrap capture wrapper targets Linux")
def test_the_neutral_wrapper_turns_alpha_off_before_it_launches_spacr(
    tmp_path, host_override, policy_status,
):
    # Exercise shell expansion and ordering without launching a namespace or
    # touching the user's preferences. The existing profile test above checks
    # the real policy's QSettings writes; this boundary records capped commands.
    repo = tmp_path / "capture repo"
    tutorials = repo / "tools" / "tutorials"
    tutorials.mkdir(parents=True)
    script = tutorials / "run_neutral_capture.sh"
    shutil.copyfile(TUTORIALS / script.name, script)
    capped = repo / "tools" / "run_capped.sh"
    capped.write_text(
        '#!/usr/bin/env bash\n'
        'printf "%s\\0" "$@" >> "$CAPTURE_COMMAND_LOG"\n'
        'printf "\\0" >> "$CAPTURE_COMMAND_LOG"\n'
        'if [[ $1 == 2G ]]; then exit "$CAPTURE_POLICY_STATUS"; fi\n'
    )
    capped.chmod(0o755)
    stage = tmp_path / "private stage"
    profile = stage / "config" / "mask profile"
    profile.mkdir(parents=True)
    command_log = tmp_path / "commands"
    capture_python = tmp_path / "namespace python"
    host_python = tmp_path / "host python"
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("SPACR_TUTORIAL_")}
    env.update(CAPTURE_COMMAND_LOG=str(command_log),
               CAPTURE_POLICY_STATUS=str(policy_status))
    if host_override:
        env["SPACR_TUTORIAL_HOST_PYTHON"] = str(host_python)

    result = subprocess.run(
        ["bash", str(script), str(stage), str(capture_python)], env=env,
        capture_output=True, text=True, timeout=10,
    )

    assert result.returncode == policy_status, result.stderr
    commands = [command.decode().split("\0")
                for command in command_log.read_bytes().split(b"\0\0") if command]
    assert commands[0] == [
        "2G", "env", "QT_QPA_PLATFORM=offscreen",
        str(host_python if host_override else capture_python),
        str(tutorials / "capture_policy.py"), "--force-alpha-off",
        str(stage / "profile" / ".config"), str(profile),
    ]
    if policy_status:
        assert len(commands) == 1, "A refused alpha policy must prevent capture"
    else:
        assert len(commands) == 2
        assert commands[1][:2] == ["6G", "bwrap"]
        assert str(capture_python) in commands[1]
        assert "tools/tutorials/capture_refresh.py" in commands[1]
        assert str(host_python) not in commands[1]


def test_capture_refresh_forces_off_before_the_window_and_passes_only_the_opt_in():
    source = (TUTORIALS / "capture_refresh.py").read_text()
    assert source.index("configure_appearance(args.theme, args.backdrop,") < \
        source.index("window = gui.MainWindow()")
    assert "'--preferences-alpha-toggle-scene', action='store_true'" in source
    # Parsed rather than matched: the call's other arguments now nest
    # parentheses. 624134261 also passes the one allow-listed alpha lesson
    # through; nothing else may reach the alpha opt-ins.
    import ast
    calls = [node for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.Call)
             and getattr(node.func, "id", None) == "verify_appearance"]
    assert len(calls) == 1
    (call,) = calls
    assert [ast.unparse(arg) for arg in call.args] == ["window"]
    keywords = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
    assert keywords["allow_alpha_toggle_scene"] == "args.preferences_alpha_toggle_scene"
    assert keywords["alpha_lesson"] == "args.alpha_lesson"
    assert not {k for k in keywords if "alpha" in k} - {
        "allow_alpha_toggle_scene", "alpha_lesson"}


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
