"""Item 569: one alpha gate, off by default, for everything built from features/future.

The maintainer's rule of 2026-09-26: "there should be a preference called
show alpha features, that is off by default ... This goes for all implemented
items on the future features list from now on."

So three things are pinned here:

* the switch is off on a fresh install, persists, and applies live -- a
  screen already open changes when Preferences closes, no restart;
* every element in ``spacr.settings.ALPHA_FEATURES`` is walked and asserted
  hidden with the switch off and shown with it on, and a hidden setting still
  reaches the run;
* a feature file in ``features/future`` whose Status says it is built,
  implemented or done in its headline registers something with the gate,
  unless it is on the exemption list below.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings                              # noqa: E402

from spacr import settings as spacr_settings                      # noqa: E402
from spacr.settings import (ALPHA_FEATURES, ALPHA_KINDS,          # noqa: E402
                            _alpha_choices, _alpha_names)

ROOT = Path(__file__).resolve().parents[2]
FUTURE = ROOT / "features" / "future"

# Future items that were built and released as ordinary features before the
# alpha rule existed. Each is exempt from the guard below; the reason is the
# same for all of them.
RELEASED_BEFORE_THE_ALPHA_RULE = {
    59: "released before the alpha rule of 2026-09-26",
    404: "released before the alpha rule of 2026-09-26",
    405: "released before the alpha rule of 2026-09-26",
    407: "released before the alpha rule of 2026-09-26",
    423: "released before the alpha rule of 2026-09-26",
    424: "released before the alpha rule of 2026-09-26",
    425: "released before the alpha rule of 2026-09-26",
    474: "released before the alpha rule of 2026-09-26",
    475: "released before the alpha rule of 2026-09-26",
    476: "released before the alpha rule of 2026-09-26",
    489: "released before the alpha rule of 2026-09-26",
    490: "released before the alpha rule of 2026-09-26",
    491: "released before the alpha rule of 2026-09-26",
    501: "released before the alpha rule of 2026-09-26",
}

BUILT = re.compile(r"\b(built|implemented|done)\b", re.IGNORECASE)

# Settings-only screens that are not in the module registry but render a form.
EXTRA_SETTINGS_HOSTS = ("timelapse",)


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    """Preferences read and written through a throwaway INI file."""
    from spacr.qt import preferences

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


def _status_line(text: str) -> str:
    """The headline of a feature file's Status: its first line.

    The headline is where an item says what state it is in. The lines under
    it narrate, and a narration mentions "done" or "implemented" about parts
    of work that is still open (370: "what done looks like"; 411: "CI wiring
    implemented" under IN PROGRESS), which is not the item being built.
    """
    for line in text.splitlines():
        if line.strip().lower().startswith("status:"):
            return line.strip()
    return ""


def _item_number(path: Path):
    match = re.match(r"^(\d+)_", path.name)
    return int(match.group(1)) if match else None


def test_the_switch_is_off_by_default_and_round_trips(prefs):
    assert prefs._get_show_alpha_features() is False
    assert prefs._is_alpha_visible() is False
    prefs._set_show_alpha_features(True)
    assert prefs._get_show_alpha_features() is True
    assert prefs._is_alpha_visible() is True
    prefs._set_show_alpha_features(False)
    assert prefs._is_alpha_visible() is False
    assert prefs._is_alpha_visible("settings", "src") is True
    assert prefs.get_show_alpha() is True


def test_the_preferences_dialog_loads_and_saves_the_switch(qtbot, prefs,
                                                           monkeypatch):
    from PySide6.QtWidgets import QDialogButtonBox, QWidget

    monkeypatch.setattr(prefs, "apply_preferences_to_app",
                        lambda *args: None)
    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    switch = dialog.findChild(QWidget, "ShowAlphaFutureFeatures")
    assert switch is not None and switch.isChecked() is False
    switch.setChecked(True)
    dialog.findChild(QDialogButtonBox).accepted.emit()
    assert prefs._get_show_alpha_features() is True


def test_every_registration_names_a_known_kind_and_a_real_thing():
    """A registration that names nothing would hide nothing and pass."""
    from spacr.qt.app import APPS

    source = "\n".join(path.read_text(encoding="utf-8", errors="ignore")
                       for path in (ROOT / "spacr").rglob("*.py")
                       if "i18n_catalogs" not in path.parts)
    for item, entry in ALPHA_FEATURES.items():
        assert entry, f"item {item} registers nothing"
        assert set(entry) <= set(ALPHA_KINDS), (item, set(entry))
        for key in entry.get("settings", ()):
            assert key in spacr_settings.expected_types, (item, key)
        for key, values in (entry.get("choices") or {}).items():
            assert key in spacr_settings.expected_types, (item, key)
            assert values, (item, key)
        for name in entry.get("widgets", ()):
            assert f'setObjectName("{name}")' in source, (item, name)
        for key in entry.get("apps", ()):
            assert key in {row[0] for row in APPS}, (item, key)


def _hosts_for(keys):
    """``app -> keys`` for the first settings screen that has each key."""
    from spacr.qt.app import APPS
    from spacr.qt.screens.settings_model import resolve_default_settings

    candidates = [row[0] for row in APPS] + list(EXTRA_SETTINGS_HOSTS)
    hosts = {}
    for key in sorted(keys):
        for app_key in candidates:
            try:
                defaults = resolve_default_settings(app_key)
            except Exception:
                continue
            if key in defaults:
                hosts.setdefault(app_key, set()).add(key)
                break
        else:
            pytest.fail(f"alpha setting {key!r} is on no settings screen; "
                        "add its screen to EXTRA_SETTINGS_HOSTS")
    return hosts


def _alpha_rows(screen, keys):
    return {key: screen.setting_row_is_visible(key) for key in keys}


def _alpha_choice_rows(screen):
    out = {}
    model = screen._settings_model
    for key in _alpha_names("choices"):
        combo = model._built_control(key)
        if combo is None:
            continue
        for index in range(combo.count()):
            if combo.itemText(index) in _alpha_choices(key):
                out[(key, combo.itemText(index))] = (
                    not combo.view().isRowHidden(index))
    return out


def test_every_registered_setting_and_choice_follows_the_switch_live(
        qtbot, prefs):
    """The registry walk: hidden off, shown on, hidden again, value kept."""
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.settings_search import ALL, install
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

    wanted = set(_alpha_names("settings")) | set(_alpha_names("choices"))
    seen_choices = set()
    for app_key, keys in _hosts_for(wanted).items():
        screen = AppScreen(app_key)
        try:
            for dimension in ("z", "t"):
                screen.set_dimension(dimension, True)
            bar = install(screen) or getattr(screen, "_settings_search", None)
            if bar is not None:
                bar.set_level(ALL)
            settings_keys = keys & _alpha_names("settings")
            for key in keys:
                screen._open_the_heading_of(key)
            screen._refresh_alpha_visibility()

            assert not any(_alpha_rows(screen, settings_keys).values()), (
                app_key, _alpha_rows(screen, settings_keys))
            assert not any(_alpha_choice_rows(screen).values()), app_key
            if bar is not None:
                assert not settings_keys & set(bar.indexed_keys())
                for key in settings_keys:
                    bar.set_query(key)
                    assert key not in bar.visible_keys()
                bar.set_query("")
            collected = screen._settings_model.collect()
            for key in settings_keys:
                assert key in collected, (app_key, key)

            prefs._set_show_alpha_features(True)
            screen._refresh_alpha_visibility()
            assert all(_alpha_rows(screen, settings_keys).values()), (
                app_key, _alpha_rows(screen, settings_keys))
            choices = _alpha_choice_rows(screen)
            assert all(choices.values()), (app_key, choices)
            seen_choices.update(choices)
            if bar is not None:
                assert settings_keys <= set(bar.indexed_keys())

            prefs._set_show_alpha_features(False)
            screen._refresh_alpha_visibility()
            assert not any(_alpha_rows(screen, settings_keys).values())
            assert not any(_alpha_choice_rows(screen).values())
        finally:
            retire_pyqtgraph_menus(screen)
            screen.close()
            screen.deleteLater()
    expected = {(key, value) for key in _alpha_names("choices")
                for value in _alpha_choices(key)}
    assert expected <= seen_choices, expected - seen_choices


def test_a_saved_alpha_entry_still_loads_and_runs_while_hidden(qtbot, prefs):
    """Hidden is not refused: a settings file that chose timeflows keeps it."""
    from spacr.qt.screens.app_screen import AppScreen

    for key in _alpha_names("choices"):
        for app_key in _hosts_for({key}):
            screen = AppScreen(app_key)
            try:
                screen._open_the_heading_of(key)
                screen._refresh_alpha_visibility()
                value = sorted(_alpha_choices(key))[0]
                assert screen._settings_model.set_value_for_key(key, value)
                assert screen._settings_model.collect()[key] == value
            finally:
                screen.close()
                screen.deleteLater()


def test_every_registered_widget_follows_the_switch(qtbot, prefs):
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import QLabel, QWidget

    root = QWidget()
    qtbot.addWidget(root)
    things = {}
    for name in sorted(_alpha_names("widgets")):
        label = QLabel(name, root)
        label.setObjectName(name)
        action = QAction(name, root)
        action.setObjectName(name)
        things[name] = (label, action)
    idle = QLabel("never shown", root)
    idle.setObjectName(next(iter(sorted(_alpha_names("widgets"))), "x"))
    idle.setVisible(False)

    prefs._apply_alpha_widgets(root)
    for name, (label, action) in things.items():
        assert label.isHidden() and not action.isVisible(), name
    prefs._set_show_alpha_features(True)
    prefs._apply_alpha_widgets(root)
    for name, (label, action) in things.items():
        assert not label.isHidden() and action.isVisible(), name
    assert idle.isHidden()


def test_the_cluster_gpu_checkbox_is_hidden_on_the_real_screen(
        qtbot, qt_theme_applied, tmp_path, prefs):
    from tests.qt.test_distributed_jobs_screen import Runner, _manager
    from spacr.qt.screens.distributed_jobs import DistributedJobsScreen
    from spacr.remote_execution import CommandResult

    manager = _manager(tmp_path, Runner(CommandResult(0, "c-1\n")))
    screen = DistributedJobsScreen(manager=manager, threaded=False,
                                   auto_poll=False)
    qtbot.addWidget(screen)
    assert prefs._apply_alpha_widgets(screen) >= 1
    assert screen._allocated_gpus.isHidden()
    prefs._set_show_alpha_features(True)
    prefs._apply_alpha_widgets(screen)
    assert not screen._allocated_gpus.isHidden()


def test_the_gpu_progress_line_stays_hidden_while_the_gate_is_shut(qtbot,
                                                                  prefs):
    from PySide6.QtWidgets import QLabel

    from spacr import _mask_workers as mw
    from spacr.qt.screens.app_screen import AppScreen

    class Host:
        _gpu_progress = QLabel()

    host = Host()
    host._gpu_progress.setObjectName("MaskGpuProgress")
    qtbot.addWidget(host._gpu_progress)
    state = {"total_batches": 2, "completed_batches": ["a"], "workers": {
        0: {"state": "running", "completed": 1, "total": 2}}}
    line = mw._progress_line("cell", state) + "\n"
    AppScreen._show_mask_gpu_progress(host, line)
    assert host._gpu_progress.isHidden()
    prefs._set_show_alpha_features(True)
    AppScreen._show_mask_gpu_progress(host, line)
    assert not host._gpu_progress.isHidden()


def test_apps_and_model_rows_go_through_the_same_gate(qtbot, prefs,
                                                     monkeypatch):
    """Whole screens and zoo rows, with a stand-in registration of each."""
    from spacr import model_zoo as mz
    from spacr.qt.app import app_is_visible
    from spacr.qt.screens.model_zoo import _model_is_alpha_hidden

    registry = dict(ALPHA_FEATURES)
    registry[99999] = {"apps": ("power",), "models": ("alpha_standin",)}
    monkeypatch.setattr(spacr_settings, "ALPHA_FEATURES", registry)
    for entry in ALPHA_FEATURES.values():
        for key in entry.get("apps", ()):
            assert app_is_visible(key) is False, key
    assert app_is_visible("power") is False
    assert app_is_visible("mask") is True
    standin = mz.ModelEntry(key="alpha_standin_v2", name="alpha_standin",
                            kind="cellpose", source="remote")
    assert _model_is_alpha_hidden(standin) is True
    prefs._set_show_alpha_features(True)
    assert app_is_visible("power") is True
    assert _model_is_alpha_hidden(standin) is False


def test_headless_runs_never_consult_the_gate():
    """The run defaults keep every alpha setting, whatever the GUI shows."""
    from spacr.settings import set_default_settings_preprocess_generate_masks

    defaults = set_default_settings_preprocess_generate_masks({})
    assert {"mask_parallel", "mask_gpu_indices", "timeflows_model"} <= set(
        defaults)
    kept = set_default_settings_preprocess_generate_masks(
        {"timelapse_mode": "timeflows", "timeflows_model": "/m.pt",
         "mask_parallel": True})
    assert kept["timelapse_mode"] == "timeflows"
    assert kept["timeflows_model"] == "/m.pt"
    assert kept["mask_parallel"] is True


def test_a_built_future_item_registers_with_the_gate():
    """The standing rule: built from features/future means alpha until promoted."""
    offenders = []
    for path in sorted(FUTURE.glob("*.txt")):
        item = _item_number(path)
        if item is None:
            continue
        status = _status_line(path.read_text(encoding="utf-8",
                                             errors="ignore"))
        if not BUILT.search(status):
            continue
        if item in ALPHA_FEATURES or item in RELEASED_BEFORE_THE_ALPHA_RULE:
            continue
        offenders.append(f"{path.name}: {status}")
    assert not offenders, (
        "future items marked built with nothing registered in "
        "spacr.settings.ALPHA_FEATURES:\n  " + "\n  ".join(offenders))


def test_the_exemption_list_names_only_future_items():
    names = {_item_number(path) for path in FUTURE.glob("*.txt")}
    assert set(RELEASED_BEFORE_THE_ALPHA_RULE) <= names
    assert not set(RELEASED_BEFORE_THE_ALPHA_RULE) & set(ALPHA_FEATURES)
