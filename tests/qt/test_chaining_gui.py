"""The chaining strip on a real module screen.

These are integration tests against the shipped :class:`AppScreen`: the strip
is installed the way the app installs it, the values it writes are read back
off the actual settings widgets, and the "continue" buttons are the ones a
user would click.

What they hold to:

* Measure's ``src`` becomes the folder Mask **registered** — asserted against
  the artifact row, and shown to vanish when the row is removed while the
  folder stays;
* a path typed into the field is pinned, survives building the screen again,
  and is never overwritten when the upstream moves — the new location appears
  as an offer with a button instead;
* the staleness row names the cause, and clears when the module is re-run;
* the next-step buttons come from :func:`spacr.ports.next_modules`, are
  pre-filled with what the run produced, and a successor that cannot run is
  disabled with its blocking reason on it;
* the factory this module registers wires the same host signals
  ``MainWindow._build_screen`` wires — checked against that method's own
  source, so the duplication cannot rot.
"""
from __future__ import annotations

import inspect
import os
import re
import sqlite3
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QPushButton

from spacr import artifacts, chaining, ports
from spacr.qt import chaining as qt_chaining
from spacr.qt.chaining import ChainingBar, install_chaining


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch, tmp_path):
    """Never read the developer's registry override or their real pins."""
    monkeypatch.delenv(artifacts.ARTIFACTS_DB_ENV, raising=False)
    monkeypatch.setenv(chaining.PIN_STATE_ENV, str(tmp_path / "pins.json"))
    chaining.pin_store(refresh=True)
    yield
    chaining.pin_store(refresh=True)


@pytest.fixture
def pins(tmp_path):
    """A pin store of this test's own, on a file it can reopen."""
    return chaining.PinStore(str(tmp_path / "pins.json"))


@pytest.fixture
def no_recent_sources(monkeypatch):
    """Silence the real QSettings so a test decides the candidate roots."""
    import spacr.qt.prefs as prefs

    roots: dict = {}
    monkeypatch.setattr(prefs, "get_last_source",
                        lambda key: roots.get(key, ""))
    monkeypatch.setattr(prefs, "get_recent_sources",
                        lambda key, limit=8: [])
    return roots


def make_plate(root: Path, *, merged: int = 2, db_tables=None,
               crops: bool = False) -> str:
    """Build a plate folder shaped the way the mask pipeline leaves one."""
    root.mkdir(parents=True, exist_ok=True)
    if merged:
        (root / "merged").mkdir(exist_ok=True)
        for index in range(merged):
            np.save(root / "merged" / f"plate1_A01_{index}.npy",
                    np.zeros((6, 6, 3), dtype=np.uint16))
    if db_tables:
        (root / "measurements").mkdir(exist_ok=True)
        connection = sqlite3.connect(root / "measurements" / "measurements.db")
        for table in db_tables:
            connection.execute(f'CREATE TABLE "{table}" (value INTEGER)')
            connection.execute(f'INSERT INTO "{table}" VALUES (1)')
        connection.commit()
        connection.close()
    if crops:
        crop_dir = root / "data" / "A01" / "cell_png"
        crop_dir.mkdir(parents=True, exist_ok=True)
        (crop_dir / "object_1.png").write_bytes(b"\x89PNG")
    return str(root)


def run_mask(root: str, **overrides):
    settings = {"src": root, "cell_channel": 0, "cell_diameter": 30}
    settings.update(overrides)
    return artifacts.register_run_outputs(
        "mask", settings, registry=artifacts.open_registry(root))


def screen_for(qapp, app_key: str, pins):
    """Build the real module screen with a chaining strip on it."""
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(app_key=app_key)
    bar = install_chaining(screen, pins=pins)
    assert bar is not None, f"{app_key} did not get a chaining strip"
    return screen, bar


def src_value(screen):
    """Read the ``src`` widget the way the settings model would."""
    return qt_chaining._widget_value(screen._settings_model._widgets["src"])


class _FakeWorker(QObject):
    """Stands in for the pipeline worker the Run button creates."""

    finished = Signal(bool)


# ===========================================================================
# 1.1 — the field fills itself from the registry
# ===========================================================================

def test_measure_opens_on_the_folder_mask_registered(qapp, tmp_path, pins,
                                                     no_recent_sources):
    root = make_plate(tmp_path / "plateA")
    produced = run_mask(root)
    merged = next(a for a in produced if a.kind == ports.MERGED_ARRAYS)
    no_recent_sources["mask"] = root

    screen, bar = screen_for(qapp, "measure", pins)

    assert src_value(screen) == root
    assert not bar._source.isHidden()
    assert merged.path in bar._source.text()
    assert "mask" in bar._source.text()
    assert not bar.isHidden()


def test_the_field_stays_empty_without_a_registry_row(qapp, tmp_path, pins,
                                                      no_recent_sources):
    """The plate folder is fully populated; only the registry row is gone."""
    root = make_plate(tmp_path / "plateA")
    produced = run_mask(root)
    registry = artifacts.open_registry(root)
    for artifact in produced:
        registry.forget(artifact)
    no_recent_sources["mask"] = root

    screen, bar = screen_for(qapp, "measure", pins)

    assert os.path.isdir(os.path.join(root, "merged"))
    assert src_value(screen) in ("path", "", None)
    assert bar._source.isHidden()


def test_classify_gets_a_list_from_measures_database(qapp, tmp_path, pins,
                                                     no_recent_sources):
    root = make_plate(tmp_path / "plateA", db_tables=("png_list", "cell"),
                      crops=True)
    run_mask(root)
    artifacts.register_run_outputs(
        "measure", {"src": root, "save_png": True},
        registry=artifacts.open_registry(root))
    no_recent_sources["measure"] = root

    screen, _bar = screen_for(qapp, "classify", pins)

    assert src_value(screen) == [root]


def test_a_module_with_no_ports_gets_no_strip(qapp, pins):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(app_key="motility")
    assert install_chaining(screen, pins=pins) is None
    assert qt_chaining.chaining_bar(screen) is None


def test_installing_twice_returns_the_same_strip(qapp, tmp_path, pins):
    screen, bar = screen_for(qapp, "measure", pins)
    assert install_chaining(screen, pins=pins) is bar


# ---------------------------------------------------------------------------
# 1.1 — the user's edit wins, and keeps winning
# ---------------------------------------------------------------------------

def test_a_typed_path_is_pinned_and_survives_rebuilding_the_screen(
        qapp, tmp_path, pins, no_recent_sources):
    masked = make_plate(tmp_path / "plateA")
    run_mask(masked)
    chosen = make_plate(tmp_path / "plateB", merged=0)
    no_recent_sources["mask"] = masked

    screen, bar = screen_for(qapp, "measure", pins)
    assert src_value(screen) == masked          # chained, not chosen

    field = screen._settings_model._widgets["src"]
    field.setText(chosen)
    field.textEdited.emit(chosen)
    bar.refresh()

    assert pins.pinned("measure", "src") == chosen

    # Reopen: a new screen and a new store reading the same file, which is
    # what surviving a restart actually means.
    reopened, reopened_bar = screen_for(
        qapp, "measure", chaining.PinStore(pins.path))

    assert src_value(reopened) == chosen
    assert reopened_bar.held["src"].value == chosen


def test_a_moved_upstream_is_offered_and_only_applied_on_request(
        qapp, tmp_path, pins, no_recent_sources):
    masked = make_plate(tmp_path / "plateA")
    run_mask(masked)
    chosen = make_plate(tmp_path / "plateB", merged=0)
    pins.pin("measure", "src", chosen)
    no_recent_sources["mask"] = masked

    screen, bar = screen_for(qapp, "measure", pins)

    assert src_value(screen) == chosen
    assert not bar._pinned_row.isHidden()
    assert chosen in bar._pinned.text()
    assert masked in bar._pinned.text()

    bar._btn_use.click()

    assert src_value(screen) == masked
    assert pins.pinned("measure", "src") is None
    assert bar._pinned_row.isHidden()


def test_a_chained_value_is_not_mistaken_for_a_user_edit(
        qapp, tmp_path, pins, no_recent_sources):
    """Filling the field must not pin it, or nothing would ever chain again."""
    masked = make_plate(tmp_path / "plateA")
    run_mask(masked)
    no_recent_sources["mask"] = masked

    screen, bar = screen_for(qapp, "measure", pins)
    bar.refresh()
    bar.refresh()

    assert src_value(screen) == masked
    assert pins.pins("measure") == {}


def test_the_first_refresh_after_a_restart_does_not_drop_a_pin(
        qapp, tmp_path, pins, no_recent_sources):
    """A placeholder is "nothing typed yet", never "the user cleared it"."""
    chosen = make_plate(tmp_path / "plateB", merged=0)
    pins.pin("measure", "src", chosen)

    screen, _bar = screen_for(qapp, "measure", chaining.PinStore(pins.path))

    assert chaining.PinStore(pins.path).pinned("measure", "src") == chosen
    assert src_value(screen) == chosen


def test_clearing_the_field_hands_the_default_back(
        qapp, tmp_path, pins, no_recent_sources):
    masked = make_plate(tmp_path / "plateA")
    run_mask(masked)
    chosen = make_plate(tmp_path / "plateB", merged=0)
    no_recent_sources["mask"] = masked
    pins.pin("measure", "src", chosen)

    screen, bar = screen_for(qapp, "measure", pins)
    assert src_value(screen) == chosen

    field = screen._settings_model._widgets["src"]
    field.setText("")
    field.textEdited.emit("")
    bar.refresh()

    assert pins.pinned("measure", "src") is None
    assert src_value(screen) == masked


def test_pressing_run_pins_whatever_is_in_the_field(qapp, tmp_path, pins,
                                                    no_recent_sources):
    chosen = make_plate(tmp_path / "plateB", merged=0)
    screen, bar = screen_for(qapp, "measure", pins)
    screen._settings_model._widgets["src"].setText(chosen)

    bar._on_run_clicked()

    assert pins.pinned("measure", "src") == chosen


def test_running_a_module_records_the_plate_for_the_next_one(qapp, tmp_path,
                                                            pins, monkeypatch):
    """The link that makes a blank downstream screen find anything at all.

    Before this, only Annotate and Make Masks called ``push_recent_source``,
    so nothing recorded which plate Mask had run on — and a Measure screen
    opened from scratch had no project whose registry it could ask.
    """
    import spacr.qt.prefs as prefs

    recorded: list = []
    monkeypatch.setattr(prefs, "push_recent_source",
                        lambda key, path, limit=8: recorded.append((key, path)))
    root = make_plate(tmp_path / "plateA")
    screen, bar = screen_for(qapp, "mask", pins)
    screen._settings_model._widgets["src"].setText(root)

    bar._on_run_clicked()

    assert recorded == [("mask", root)]


def test_a_blank_downstream_screen_finds_the_plate_the_upstream_ran_on(
        qapp, tmp_path, pins, no_recent_sources):
    """Mask runs on a plate; Measure, opened cold, offers that plate."""
    root = make_plate(tmp_path / "plateA")
    mask_screen, mask_bar = screen_for(qapp, "mask", pins)
    mask_screen._settings_model._widgets["src"].setText(root)
    # Standing in for the QSettings write ``_remember_project`` makes.
    no_recent_sources["mask"] = mask_bar._remember_project() or root
    run_mask(root)

    measure_screen, _bar = screen_for(qapp, "measure", pins)

    assert src_value(measure_screen) == root


# ===========================================================================
# 1.2 — staleness, where the user is about to act
# ===========================================================================

def _register_measure_run(screen, root):
    """Register a Measure run using exactly the settings on this screen."""
    settings = dict(screen._settings_model.collect())
    return artifacts.register_run_outputs(
        "measure", settings, registry=artifacts.open_registry(root))


def test_an_upstream_re_run_shows_up_on_the_screen_with_its_cause(
        qapp, tmp_path, pins, no_recent_sources):
    root = make_plate(tmp_path / "plateA", db_tables=("png_list", "cell"))
    run_mask(root)
    no_recent_sources["mask"] = root

    screen, bar = screen_for(qapp, "measure", pins)
    assert src_value(screen) == root
    _register_measure_run(screen, root)
    bar.refresh()
    assert bar._stale.isHidden()

    time.sleep(0.01)
    np.save(Path(root) / "merged" / "extra.npy",
            np.zeros((6, 6, 3), dtype=np.uint16))
    run_mask(root)
    bar.refresh()

    assert not bar._stale.isHidden()
    notes = bar.stale_notes()
    assert artifacts.CAUSE_UPSTREAM_SUPERSEDED in notes[0].causes
    assert "newer run has replaced one of its inputs" in bar._stale.text()
    assert not bar._fix.isHidden()
    assert "Re-run measure" in bar._fix.text()

    # …and clears when Measure runs again.
    time.sleep(0.01)
    connection = sqlite3.connect(
        Path(root) / "measurements" / "measurements.db")
    connection.execute('INSERT INTO "cell" VALUES (2)')
    connection.commit()
    connection.close()
    _register_measure_run(screen, root)
    bar.refresh()

    assert bar.stale_notes() == ()
    assert bar._stale.isHidden()


def test_changing_a_setting_marks_the_existing_result_stale(
        qapp, tmp_path, pins, no_recent_sources):
    root = make_plate(tmp_path / "plateA", db_tables=("png_list", "cell"))
    run_mask(root)
    no_recent_sources["mask"] = root

    screen, bar = screen_for(qapp, "measure", pins)
    _register_measure_run(screen, root)
    bar.refresh()
    assert bar._stale.isHidden()

    screen._settings_model._widgets["cell_mask_dim"].setValue(5)
    bar.refresh()

    notes = bar.stale_notes()
    assert notes
    assert artifacts.CAUSE_SETTINGS_CHANGED in notes[0].causes
    assert "settings on this screen differ" in bar._stale.text()
    assert "Re-run measure with these settings" in bar._fix.text()


def test_the_strip_sits_immediately_above_the_run_button(qapp, tmp_path,
                                                         pins):
    """Where the user is about to act, not in a panel they would have to open."""
    screen, bar = screen_for(qapp, "measure", pins)
    layout = screen._runtime_wrap.layout()

    assert layout.indexOf(bar) == layout.indexOf(screen._actions_row) - 1
    assert screen._btn_run.parent() is screen._actions_row


def test_the_strip_hides_itself_when_there_is_nothing_to_say(qapp, tmp_path,
                                                             pins):
    screen, bar = screen_for(qapp, "measure", pins)
    bar.refresh()
    assert bar.isHidden()


# ===========================================================================
# 1.3 — continue to the next step
# ===========================================================================

def test_a_finished_mask_run_offers_measure(qapp, tmp_path, pins,
                                            no_recent_sources):
    root = make_plate(tmp_path / "plateA")
    run_mask(root)

    screen, bar = screen_for(qapp, "mask", pins)
    screen._settings_model._widgets["src"].setText(root)
    bar._on_run_finished(True)

    assert [s.module for s in bar.steps] == list(ports.next_modules("mask"))
    assert not bar._next_row.isHidden()
    buttons = bar.findChildren(QPushButton)
    measure = [b for b in buttons if b.text() == "Measure"]
    assert measure and measure[0].isEnabled()
    assert bar.steps[0].seed["src"] == root


def test_an_unready_successor_is_shown_with_its_reason(qapp, tmp_path, pins,
                                                       no_recent_sources):
    """Measure wrote no ``png_list``, so Classify is offered but disabled."""
    root = make_plate(tmp_path / "plateA", db_tables=("cell",))
    run_mask(root)
    artifacts.register_run_outputs(
        "measure", {"src": root}, registry=artifacts.open_registry(root))

    screen, bar = screen_for(qapp, "measure", pins)
    screen._settings_model._widgets["src"].setText(root)
    bar._on_run_finished(True)

    blocked = [s for s in bar.steps if s.module == "classify"]
    assert blocked and not blocked[0].ok
    button = next(b for b in bar.findChildren(QPushButton)
                  if b.text().startswith("Classify (CV)"))
    assert not button.isEnabled()
    assert "not ready" in button.text()
    assert "png_list" in button.toolTip()
    assert blocked[0].fix in button.toolTip()


def test_a_failed_run_offers_nothing(qapp, tmp_path, pins):
    root = make_plate(tmp_path / "plateA")
    run_mask(root)
    screen, bar = screen_for(qapp, "mask", pins)
    screen._settings_model._widgets["src"].setText(root)

    bar._on_run_finished(False)

    assert bar.steps == ()
    assert bar._next_row.isHidden()


def test_the_run_button_hooks_the_worker_that_finishes(qapp, tmp_path, pins,
                                                       no_recent_sources):
    """The strip follows the run without a line inside AppScreen."""
    root = make_plate(tmp_path / "plateA")
    run_mask(root)
    screen, bar = screen_for(qapp, "mask", pins)
    screen._settings_model._widgets["src"].setText(root)

    worker = _FakeWorker()
    screen._worker = worker
    bar._on_run_clicked()
    worker.finished.emit(True)

    assert [s.module for s in bar.steps] == ["measure"]


def test_continue_navigates_and_seeds_without_pinning(qapp, tmp_path, pins,
                                                      no_recent_sources,
                                                      monkeypatch):
    """The seed is an artifact the registry resolved, so it is not an edit."""
    root = make_plate(tmp_path / "plateA")
    run_mask(root)

    mask_screen, mask_bar = screen_for(qapp, "mask", pins)
    mask_screen._settings_model._widgets["src"].setText(root)
    measure_screen, measure_bar = screen_for(qapp, "measure", pins)

    opened: list = []

    class _Window:
        _screens = {"measure": measure_screen}

        def _on_nav_selected(self, key):
            opened.append(key)

    window = _Window()
    monkeypatch.setattr(type(mask_bar), "host_window", lambda self: window)

    mask_bar._on_run_finished(True)
    mask_bar._on_continue(mask_bar.steps[0])

    assert opened == ["measure"]
    assert src_value(measure_screen) == root
    assert pins.pins("measure") == {}


# ===========================================================================
# Installation seams
# ===========================================================================

def test_the_factory_wires_the_same_signals_build_screen_wires():
    """The duplicated wiring is checked against its original, not trusted."""
    from spacr.qt.app import MainWindow

    source = inspect.getsource(MainWindow._build_screen)
    tail = source.split("from .screens.app_screen import AppScreen")[-1]
    wired = dict(re.findall(r"screen\.(\w+)\.connect\(\s*self\.(\w+)", tail))

    assert wired, "no AppScreen signal connections found to compare against"
    assert wired == qt_chaining.HOST_CONNECTIONS


def test_no_chained_module_has_its_own_branch_in_build_screen():
    """The factory only pre-empts the generic AppScreen tail."""
    from spacr.qt.app import MainWindow

    source = inspect.getsource(MainWindow._build_screen)
    branch_keys = set(re.findall(r'if key == "([^"]+)"', source))

    assert branch_keys, "no explicit screen branches found"
    assert branch_keys.isdisjoint(qt_chaining.chained_app_keys())


def test_register_installs_a_factory_for_every_ported_app():
    from spacr.qt.app import APP_FACTORIES, APPS

    before = dict(APP_FACTORIES)
    try:
        assert qt_chaining.register() is True
        keys = qt_chaining.chained_app_keys()
        assert "measure" in keys and "mask" in keys
        assert set(keys) <= {row[0] for row in APPS}
        for key in keys:
            assert APP_FACTORIES[key] is qt_chaining._chained_app_screen
        # Idempotent.
        assert qt_chaining.register() is True
        assert qt_chaining.unregister() == len(keys)
    finally:
        APP_FACTORIES.clear()
        APP_FACTORIES.update(before)


def test_register_does_not_steal_a_screen_somebody_else_owns():
    from spacr.qt.app import APP_FACTORIES

    before = dict(APP_FACTORIES)
    try:
        mine = object()
        APP_FACTORIES["measure"] = mine
        qt_chaining.register()
        assert APP_FACTORIES["measure"] is mine
    finally:
        APP_FACTORIES.clear()
        APP_FACTORIES.update(before)


def test_the_chaining_module_is_named_by_the_launch_seam():
    """``spacr.qt.run`` imports this module between app.py and the window.

    :data:`spacr.qt.SELF_REGISTERING_MODULES` is where a module asks to be
    imported at that moment — after ``app.py`` is loaded, so ``APP_FACTORIES``
    exists, and before ``MainWindow.__init__``, so the first screen already
    has its strip.  The list itself belongs to another workstream and lands on
    its own, so this asserts the entry when the seam is present rather than
    turning red on a checkout that has the strip but not yet the list.
    """
    import spacr.qt as qt

    modules = getattr(qt, "SELF_REGISTERING_MODULES", None)
    if modules is None:
        pytest.skip("the launch self-registration seam is not in this build")
    assert "spacr.qt.chaining" in modules


def test_the_factory_builds_a_screen_with_a_strip(qapp):
    screen = qt_chaining._chained_app_screen("measure")
    try:
        assert qt_chaining.chaining_bar(screen) is not None
        assert screen.app_key == "measure"
    finally:
        screen.deleteLater()


def test_the_strip_never_takes_the_screen_down(qapp, tmp_path, pins,
                                               monkeypatch):
    """A registry that explodes costs the strip, never the module screen."""
    def boom(*_args, **_kwargs):
        raise RuntimeError("registry on fire")

    screen, bar = screen_for(qapp, "measure", pins)
    monkeypatch.setattr(chaining, "resolve_settings", boom)

    bar.refresh()                      # must not raise

    monkeypatch.setattr(qt_chaining, "ChainingBar", boom)
    from spacr.qt.screens.app_screen import AppScreen
    other = AppScreen(app_key="measure")
    assert install_chaining(other, pins=pins) is None


def test_the_stylesheet_block_is_registered_and_renders():
    from spacr.qt.app import APP_FACTORIES
    from spacr.qt.theme import (DARK_PALETTE, registered_widget_qss,
                                stylesheet, widget_qss_names)

    before = dict(APP_FACTORIES)
    try:
        qt_chaining.register()
        assert "ChainingBar" in widget_qss_names()
        assert "QFrame#ChainingBar" in registered_widget_qss(dict(DARK_PALETTE))
        rendered = qt_chaining._qss(dict(DARK_PALETTE, theme="dark",
                                         font_scale=1.0), None)
        assert "QFrame#ChainingBar" in rendered
        assert "ChainingBar" in stylesheet()
    finally:
        APP_FACTORIES.clear()
        APP_FACTORIES.update(before)


def test_every_module_screen_still_builds_with_the_factories_installed(qapp):
    """The production path: ``register()`` has run, then a window opens.

    ``MainWindow._build_screen`` is called the way the module smoke test calls
    it — unbound, against a stand-in host — so a factory that dropped a signal
    or refused a key would fail here rather than at launch.
    """
    from spacr.qt.app import APP_FACTORIES, MainWindow
    from spacr.qt.screens.app_screen import AppScreen
    from .test_all_module_smoke import _FactoryHost

    before = dict(APP_FACTORIES)
    try:
        qt_chaining.register()
        host = _FactoryHost()
        for key in qt_chaining.chained_app_keys():
            screen = MainWindow._build_screen(host, key)
            try:
                assert isinstance(screen, AppScreen)
                assert screen.app_key == key
                assert qt_chaining.chaining_bar(screen) is not None
            finally:
                screen.deleteLater()
    finally:
        APP_FACTORIES.clear()
        APP_FACTORIES.update(before)
