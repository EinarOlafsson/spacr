"""A recruitment settings file from before 79edbf12f still loads.

Instruction 364 took `cell_mask_dim`, `nucleus_mask_dim` and
`pathogen_mask_dim` out of recruitment on 2026-09-03 (79edbf12f): the factory
stopped writing them and the form stopped showing them, because
`analyze_recruitment` never read any of the three. Every recruitment settings
CSV saved before that date still names them.

They were NOT added to `validate.RETIRED_SETTINGS`, and that is deliberate.
Measure and the merged-plot settings still read the same three keys, so a
global retirement would warn a Measure user off a setting that works.

So an old recruitment file has no rename to follow. What it must do instead,
tested one key at a time the way a user meets it (Import settings on the
Recruitment screen, in a real window):

* the file loads without rebuilding the screen, and every value the form
  still carries is applied. Until 2026-09-19 `nucleus_mask_dim` and
  `pathogen_mask_dim` rebuilt the screen without end: the import took
  them for object switches, and a form that no longer carries them could
  never agree with the file;
* the old mask plane goes nowhere: no hidden form value and no key in what
  the run is handed, because nothing on this path would read it;
* the pre-flight check says nothing about it -- the key is a live Measure
  setting, so neither "renamed" nor "no longer a spaCR setting" is true;
* today's factory does not put it back.
"""
from __future__ import annotations

import csv
import inspect
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

MASK_DIMS = ("cell_mask_dim", "nucleus_mask_dim", "pathogen_mask_dim")

#: What `get_analyze_recruitment_default_settings` wrote for the three before
#: 79edbf12f, and what a saved recruitment file therefore carries.
OLD_DEFAULTS = {"cell_mask_dim": 4, "nucleus_mask_dim": 5,
                "pathogen_mask_dim": 6}


def _old_recruitment_file(tmp_path, key):
    """A settings CSV as a pre-2026-09-03 recruitment run saved it.

    ``cell_chann_dim`` and ``channel_of_interest`` are set away from their
    defaults so the test can tell a file that loaded from one that did not.
    """
    path = tmp_path / "recruitment_settings.csv"
    rows = [("Key", "Value"), ("src", str(tmp_path)),
            ("cell_chann_dim", "1"), ("channel_of_interest", "1"),
            (key, str(OLD_DEFAULTS[key])),
            ("nucleus_chann_dim", "0"), ("pathogen_chann_dim", "2")]
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    return path


def test_the_recruitment_run_reads_none_of_the_three():
    """The premise, re-measured rather than quoted from 2026-09-03."""
    from spacr.submodules import analyze_recruitment

    source = inspect.getsource(analyze_recruitment)
    for key in MASK_DIMS:
        assert key not in source, f"analyze_recruitment reads {key} now"


def test_the_three_are_still_live_settings_elsewhere():
    """Why there is no RETIRED_SETTINGS entry: Measure reads them."""
    from spacr.settings import expected_types, get_measure_crop_settings
    from spacr.validate import RETIRED_SETTINGS

    measure = get_measure_crop_settings({})
    for key in MASK_DIMS:
        assert key in expected_types, key
        assert key in measure, f"{key} left Measure too"
        assert key not in RETIRED_SETTINGS, (
            f"{key} is live in Measure; retiring it globally would warn a "
            "Measure user off a setting that works")


@pytest.mark.parametrize("key", MASK_DIMS)
def test_todays_recruitment_factory_does_not_write_it(key):
    from spacr.settings import get_analyze_recruitment_default_settings

    assert key not in get_analyze_recruitment_default_settings({})


@pytest.mark.parametrize("key", MASK_DIMS)
def test_the_doctor_says_nothing_about_it(tmp_path, key):
    """No "renamed", no "no longer a spaCR setting", no typo guess."""
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.validate import validate_settings

    loaded = AppScreen._load_settings_csv(str(_old_recruitment_file(
        tmp_path, key)))
    assert str(loaded.get(key)) == str(OLD_DEFAULTS[key])

    about_it = [p for p in validate_settings(loaded, "recruitment")
                if p.setting == key]
    assert about_it == [], [p.message for p in about_it]


#: How many screen rebuilds one import may cost before the test stops it.
#: Before the fix an old file carrying `nucleus_mask_dim` or
#: `pathogen_mask_dim` rebuilt the screen about once a second without end,
#: so an uncapped run of this test would never return.
REBUILD_CAP = 3


@pytest.fixture
def recruitment_window(qtbot, qt_theme_applied, monkeypatch):
    """A real window with Recruitment open, rebuilds counted and capped.

    A screen built on its own has no window to ask for a rebuild, which is
    how the first version of this test passed over a file that hung the
    app: the import rebuilt the screen forever and only a MainWindow can.
    """
    from spacr.qt.app import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1400, 900)
    window.show()
    qtbot.waitExposed(window)
    assert window.open_module("recruitment") == "recruitment"
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


def _import_through_the_button(window, monkeypatch, path):
    """Press Import settings on the open screen and choose ``path``.

    :returns: the ``Import failed`` warnings the screen raised.
    """
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(path), "")))
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: warnings.append(a[1:])))
    window._screens["recruitment"]._on_import_settings()
    return warnings


@pytest.mark.parametrize("key", MASK_DIMS)
def test_importing_an_old_file_on_the_recruitment_screen(
        recruitment_window, tmp_path, monkeypatch, key):
    """Import settings, as the button does it, in the real window."""
    window, rebuilds = recruitment_window
    model = window._screens["recruitment"]._settings_model
    assert key not in model._widgets, f"{key} is back on the form"
    before = dict(model.collect())
    assert before.get("cell_chann_dim") != 1, (
        "pick a probe unlike the default or this test proves nothing")

    warnings = _import_through_the_button(
        window, monkeypatch, _old_recruitment_file(tmp_path, key))

    assert warnings == [], f"the import failed: {warnings}"
    assert rebuilds == [], (
        f"a key the form does not carry rebuilt the screen {len(rebuilds)} "
        "times; it cannot change the form's shape")
    after = dict(window._screens["recruitment"]._settings_model.collect())
    assert after["cell_chann_dim"] == 1, "the file did not load"
    assert after["channel_of_interest"] == 1, "the file did not load"
    assert key not in after, (
        f"{key} reached the run's settings; nothing on recruitment reads it")
