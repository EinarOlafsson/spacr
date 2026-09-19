"""Importing a settings file rebuilds the screen at most once.

`AppScreen.apply_settings_dict` rebuilds the screen when the file changes a
value that shapes the form -- the organelle count, or an object's channel or
mask plane -- and then applies the file to the new screen. Until 2026-09-19
it treated every `nucleus_*` / `pathogen_*` channel or mask-plane key in the
FILE as one of those, whether or not the form carried it. A key the form
does not carry is never in what the form collects, so the new screen
disagreed with the file exactly as the old one had, and asked for another
rebuild, and so on without end. Measured through a real window before the
fix, with the rebuild capped at six: every case below hit the cap and ended
in "Import failed"; uncapped it runs until something kills it.

* an old recruitment file (see
  `test_an_old_recruitment_file_with_mask_dims_still_loads.py`);
* a Mask settings file imported on Measure, which has no `*_channel`;
* a Measure settings file imported on Mask, which has no `*_mask_dim`.

The same dict is meant to apply safely across apps (the method's own
docstring says so), so the last two are the ordinary case, not an edge.

Two things are pinned. A switch the form does not carry shapes nothing,
which is the rule `_form_shaping_keys` already followed for a typed edit.
And whatever the shape check says, the screen a rebuild produced is built
for the file and is never rebuilt again by the same import.
"""
from __future__ import annotations

import csv

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

#: Past this many rebuilds for one import the counting wrapper raises,
#: which the import reports as "Import failed". The pre-fix behaviour
#: would otherwise never return.
REBUILD_CAP = 3


def _open(qtbot, monkeypatch, app_key):
    """A real window with ``app_key`` open and its rebuilds counted."""
    from spacr.qt.app import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1400, 900)
    window.show()
    qtbot.waitExposed(window)
    assert window.open_module(app_key) == app_key
    qtbot.wait(20)

    rebuilds = []
    real = window.rebuild_app_screen

    def counted(key, values=None):
        rebuilds.append(key)
        if len(rebuilds) > REBUILD_CAP:
            raise RuntimeError(f"{len(rebuilds)} rebuilds for one import")
        return real(key, values)

    monkeypatch.setattr(window, "rebuild_app_screen", counted)
    return window, rebuilds


def _import(window, monkeypatch, tmp_path, app_key, rows):
    """Write ``rows`` as a settings CSV and import it with the button.

    :returns: the warnings the screen raised.
    """
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    path = tmp_path / f"{app_key}_settings.csv"
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows([("Key", "Value")] + list(rows))
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(path), "")))
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: warnings.append(a[1:])))
    window._screens[app_key]._on_import_settings()
    return warnings


def _collected(window, app_key):
    return dict(window._screens[app_key]._settings_model.collect())


@pytest.mark.parametrize("app_key, absent, probe", [
    ("measure", {"nucleus_channel": "0", "pathogen_channel": "2"},
     "cell_min_size"),
    ("mask", {"nucleus_mask_dim": "5", "pathogen_mask_dim": "6"},
     "cell_background"),
], ids=["a Mask file on Measure", "a Measure file on Mask"])
def test_another_modules_switches_do_not_rebuild_the_screen(
        qtbot, qt_theme_applied, monkeypatch, tmp_path, app_key, absent,
        probe):
    window, rebuilds = _open(qtbot, monkeypatch, app_key)
    model = window._screens[app_key]._settings_model
    for key in absent:
        assert key not in model._widgets, (
            f"{app_key} carries {key} now, so this case no longer probes a "
            "switch the form lacks")
    assert _collected(window, app_key).get(probe) != 321, (
        "pick a probe unlike the default or this test proves nothing")

    warnings = _import(window, monkeypatch, tmp_path, app_key,
                       list(absent.items()) + [(probe, "321")])

    assert warnings == [], f"the import failed: {warnings}"
    assert rebuilds == [], (
        f"{len(rebuilds)} rebuilds for switches {app_key} does not carry")
    assert _collected(window, app_key)[probe] == 321, (
        "the file's other values did not load")


def test_a_switch_the_form_carries_still_lands(
        qtbot, qt_theme_applied, monkeypatch, tmp_path):
    """The filter must not cost a real switch its value."""
    window, rebuilds = _open(qtbot, monkeypatch, "measure")
    model = window._screens["measure"]._settings_model
    assert "nucleus_mask_dim" in model._widgets
    assert _collected(window, "measure").get("nucleus_mask_dim") != 7

    warnings = _import(window, monkeypatch, tmp_path, "measure",
                       [("nucleus_mask_dim", "7")])

    assert warnings == [], f"the import failed: {warnings}"
    assert len(rebuilds) <= 1, f"{len(rebuilds)} rebuilds for one import"
    assert _collected(window, "measure")["nucleus_mask_dim"] == 7


def test_the_organelle_count_still_rebuilds_once_and_spawns_its_slot(
        qtbot, qt_theme_applied, monkeypatch, tmp_path):
    """The one change that adds rows still gets its rebuild, and only one."""
    window, rebuilds = _open(qtbot, monkeypatch, "mask")
    assert _collected(window, "mask").get("number_of_organelles") == 0

    warnings = _import(window, monkeypatch, tmp_path, "mask",
                       [("number_of_organelles", "2"),
                        ("organelleb_channel", "3")])

    assert warnings == [], f"the import failed: {warnings}"
    assert rebuilds == ["mask"], rebuilds
    after = _collected(window, "mask")
    assert after["number_of_organelles"] == 2
    assert after["organelleb_channel"] == 3


def test_the_screen_a_rebuild_made_is_not_rebuilt_again(
        qtbot, qt_theme_applied, monkeypatch, tmp_path):
    """The guard, independent of which keys the shape check counts.

    The replacement screen is built from the file's merged values, so a
    second rebuild would build the same screen again. Forcing the shape
    check to answer yes every time is the general form of every loop above.
    """
    from spacr.qt.screens.app_screen import AppScreen

    window, rebuilds = _open(qtbot, monkeypatch, "measure")
    monkeypatch.setattr(AppScreen, "_bulk_apply_changes_form_shape",
                        lambda self, settings, current: True)

    warnings = _import(window, monkeypatch, tmp_path, "measure",
                       [("cell_min_size", "321")])

    assert warnings == [], f"the import failed: {warnings}"
    assert rebuilds == ["measure"], rebuilds
    assert _collected(window, "measure")["cell_min_size"] == 321

    rebuilds.clear()
    _import(window, monkeypatch, tmp_path, "measure",
            [("cell_min_size", "123")])
    assert rebuilds == ["measure"], (
        "the guard outlived its import: a later import on the rebuilt "
        f"screen could not rebuild it ({rebuilds})")
