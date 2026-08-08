"""Tests for the "Load demo dataset…" menu action in MainWindow.

We bypass the QFileDialog by calling `_run_demo_generator` +
`_apply_demo_to_screen` directly with a tmp_path destination — that's
what the menu callback would call after the user picks a folder.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _new_mainwindow(qtbot, qt_theme_applied):
    from spacr.qt.app import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)
    return win


def test_demo_targets_cover_every_generator(qtbot, qt_theme_applied):
    """Every DEMO_TARGETS entry must reference a real generator on
    spacr.qt.synthetic — a typo here would silently break the menu."""
    from spacr.qt import synthetic as syn
    win = _new_mainwindow(qtbot, qt_theme_applied)
    for demo_key, (target_app, gen_name) in win.DEMO_TARGETS.items():
        assert hasattr(syn, gen_name), (
            f"DEMO_TARGETS['{demo_key}'] points at missing "
            f"spacr.qt.synthetic.{gen_name}")


@pytest.mark.parametrize("demo_key", ["mask", "measure", "crop",
                                        "classify", "timelapse",
                                        "map_barcodes"])
def test_run_demo_generator_produces_layout(qtbot, qt_theme_applied,
                                              tmp_path, demo_key):
    win = _new_mainwindow(qtbot, qt_theme_applied)
    layout = win._run_demo_generator(demo_key, str(tmp_path))
    assert layout.src.exists()
    # Every generator returns a settings CSV that spacr.utils can read
    assert layout.settings_csv is not None
    assert layout.settings_csv.exists()


# The pre-flight app key each demo's settings are validated against. It is not
# always DEMO_TARGETS' target *screen*: the classify demo opens in Annotate
# (a GUI-only screen with no pipeline), and the crop demo is measure with
# save_png on.
_DEMO_PREFLIGHT_APP = {
    "mask": "mask",
    "measure": "measure",
    "crop": "measure",
    "classify": "classify",
    "timelapse": "timelapse",
    "map_barcodes": "map_barcodes",
}


@pytest.mark.parametrize("demo_key", sorted(_DEMO_PREFLIGHT_APP))
def test_every_demo_clears_preflight(qtbot, qt_theme_applied, tmp_path,
                                       demo_key):
    """Generate each demo through the menu's own code path and run spaCR's
    pre-flight on the settings it wrote.

    This is the test that would have caught all four of the demo defects at
    once: no ``merged/`` for measure, a timelapse dataset advertising channels
    it had not acquired, a misspelled ``cell_signal_to_noise``, and a
    sequencing demo with no barcode CSVs. Checking the generator in isolation
    caught none of them — the settings CSV is what the user actually loads.
    """
    from spacr.utils import load_settings
    from spacr.validate import validate_settings

    win = _new_mainwindow(qtbot, qt_theme_applied)
    layout = win._run_demo_generator(demo_key, str(tmp_path / demo_key))
    settings = load_settings(str(layout.settings_csv),
                             setting_key="Key", setting_value="Value")
    problems = validate_settings(settings, _DEMO_PREFLIGHT_APP[demo_key])
    assert not problems, (
        f"{demo_key} demo does not clear pre-flight:\n"
        + "\n".join(str(p) for p in problems))


@pytest.mark.parametrize("demo_key", sorted(_DEMO_PREFLIGHT_APP))
def test_each_demo_folder_holds_exactly_one_settings_csv(qtbot,
                                                           qt_theme_applied,
                                                           tmp_path, demo_key):
    """A demo folder must not offer the user a choice of settings files.

    "Import settings…" is a file picker pointed at the demo folder. The crop
    demo used to leave two — ``settings_measure.csv`` (``save_png=False``) and
    ``settings_crop.csv`` (``save_png=True``) — because its generator called
    ``generate_measure_demo`` for the dataset and then only reassigned
    ``layout.settings_csv``. Picking the first one gives a Crop run that writes
    no PNG crops at all, and nothing anywhere says why.

    Asserted through the menu's own code path and over *every* demo, because
    the same "reuse another generator, then rename the field" shortcut is the
    obvious way to add the next one.
    """
    win = _new_mainwindow(qtbot, qt_theme_applied)
    dst = tmp_path / demo_key
    layout = win._run_demo_generator(demo_key, str(dst))
    found = sorted(p.name for p in Path(layout.src).glob("settings*.csv"))
    assert found == [Path(layout.settings_csv).name], (
        f"{demo_key} demo folder holds {found}; the demo menu reports "
        f"{Path(layout.settings_csv).name} and a user importing either of the "
        "others gets a different run than the one the demo describes")


def test_apply_mask_demo_populates_app_screen(qtbot, qt_theme_applied,
                                                 tmp_path):
    """The mask demo → mask AppScreen path: after applying, the
    settings model's src widget should carry our tmp_path."""
    win = _new_mainwindow(qtbot, qt_theme_applied)
    layout = win._run_demo_generator("mask", str(tmp_path))

    win._on_nav_selected("mask")
    screen = win._screens.get("mask")
    assert screen is not None
    win._apply_demo_to_screen(screen, layout)

    src_w = getattr(screen, "_settings_model", None)
    assert src_w is not None
    widgets = src_w._widgets
    if "src" in widgets:
        from PySide6.QtWidgets import QLineEdit
        w = widgets["src"]
        # Whatever widget type is used for src, its value should carry
        # the tmp_path we generated the demo into
        if isinstance(w, QLineEdit):
            assert str(layout.src) in w.text()


@pytest.mark.parametrize("demo_key", sorted(_DEMO_PREFLIGHT_APP))
def test_demo_settings_survive_the_widget_round_trip(qtbot, qt_theme_applied,
                                                       tmp_path, demo_key):
    """Every demo key the generator writes has to land in a real widget
    *and come back out with the same value*.

    `_apply_demo_to_screen` loads the CSV and calls `apply_settings_dict`,
    which silently ignores keys the screen has no widget for. Asserting on the
    generator's dict alone cannot see that; asserting on what the screen holds
    afterwards can.

    Presence alone is not enough, and that is the point of the value half.
    ``normalize=[1, 99]`` was PRESENT in `collect()` and came back as
    ``False``: `spacr.settings` declares ``normalize`` a bool, so the Measure
    screen renders a Toggle, and `AppScreen._apply_value` sets it from
    ``str(val).lower() in ("true", "1", "yes")``. The CSV on disk and the form
    the user is about to hit Run on disagreed about how every crop is scaled,
    and a keys-only assertion called that green.

    The screen a demo is round-tripped against is the app its settings CSV is
    written *for* (``_DEMO_PREFLIGHT_APP``), not `DEMO_TARGETS`' navigation
    target: the classify demo opens the Annotate screen for labelling, but
    ``settings_classify.csv`` is a Classify settings file and that is the form
    it has to survive.
    """
    win = _new_mainwindow(qtbot, qt_theme_applied)
    layout = win._run_demo_generator(demo_key, str(tmp_path / demo_key))
    target_app = _DEMO_PREFLIGHT_APP[demo_key]
    win._on_nav_selected(target_app)
    screen = win._screens.get(target_app)
    assert screen is not None
    assert hasattr(screen, "apply_settings_dict"), target_app

    win._apply_demo_to_screen(screen, layout)
    collected = screen._settings_model.collect()

    from spacr.utils import load_settings
    written = load_settings(str(layout.settings_csv),
                            setting_key="Key", setting_value="Value")
    # A key a module deliberately never shows is not one it dropped. The
    # demo CSVs still carry superseded aliases -- `png_type` is the one
    # that surfaced this -- and those keys are read by the pipeline for
    # backward compatibility while being deliberately absent from the
    # form. See `_APP_HIDDEN_KEYS`, and INVARIANTS 6 for why hidden and
    # absent are different things.
    from spacr.qt.screens.settings_model import _APP_HIDDEN_KEYS

    never_shown = _APP_HIDDEN_KEYS.get(target_app, set())
    missing = sorted(k for k in written
                     if k not in collected and k not in never_shown)
    assert not missing, (
        f"{demo_key}: the {target_app} screen has no widget for "
        f"{missing} — those settings are dropped on the floor")

    # `src` is exempt from equality and checked by containment below: the
    # multi-plate apps (classify) edit it as a list of roots, so one path in
    # legitimately comes back as a one-element list. Nothing else may change.
    # `k in collected` as well: a deliberately hidden key has no widget to
    # round-trip through, so asking what the form holds for it raises
    # rather than reporting a mismatch.
    mangled = {k: (v, collected[k]) for k, v in written.items()
               if k != "src" and k in collected and collected[k] != v}
    assert not mangled, (
        f"{demo_key}: the {target_app} screen changed these values on the way "
        f"through its widgets:\n"
        + "\n".join(f"  {k}: wrote {w!r} ({type(w).__name__}), form holds "
                    f"{g!r} ({type(g).__name__})"
                    for k, (w, g) in sorted(mangled.items()))
        + "\n\nA demo that writes a setting the GUI then rewrites is worse "
          "than a demo that omits it: the CSV and the form disagree and the "
          "run uses the form.")
    assert str(layout.src) in str(collected["src"])


def test_apply_classify_demo_opens_annotate_screen(qtbot,
                                                     qt_theme_applied,
                                                     tmp_path):
    """The classify demo routes to the AnnotateScreen (not an AppScreen)
    because that's where users label the crops."""
    win = _new_mainwindow(qtbot, qt_theme_applied)
    layout = win._run_demo_generator("classify", str(tmp_path))
    win._on_nav_selected("annotate")
    screen = win._screens.get("annotate")
    assert screen is not None
    # Sanity: AnnotateScreen exposes _open_source, which is what
    # _apply_demo_to_screen falls through to when settings can't apply
    assert hasattr(screen, "_open_source")


def test_the_classify_demo_actually_opens_its_crops_in_annotate(
        qtbot, qt_theme_applied, tmp_path):
    """The handoff, not just the route.

    Landing on the Annotate screen is not the outcome a user cares about;
    seeing the 64 crops is. `_apply_demo_to_screen` hands AnnotateScreen the
    demo ROOT, and the screen goes looking for ``measurements/measurements.db``
    and a ``png_list`` table under it — so the generator's folder layout and
    the screen's expectation have to agree, and neither of them is checked by
    the layout/pre-flight tests above.
    """
    win = _new_mainwindow(qtbot, qt_theme_applied)
    layout = win._run_demo_generator("classify", str(tmp_path / "classify"))
    win._on_nav_selected("annotate")
    screen = win._screens.get("annotate")
    screen._settings.grid_rows = 2
    screen._settings.grid_cols = 2
    screen._rebuild_grid()

    win._apply_demo_to_screen(screen, layout)
    try:
        qtbot.waitUntil(lambda: len(screen._page_paths) == 4, timeout=10000)
        # Every crop the generator wrote is reachable through the screen.
        assert screen._total == 64, screen._total
        assert screen._settings.db_path == str(layout.db_path)
        for path, _label in screen._page_paths:
            assert Path(path).is_file(), path
        # Unlabelled to start with: the demo's `annotate` column is what the
        # Classify run trains on, and the user is here to set it.
        assert screen._settings.annotation_column == "annotate"
    finally:
        if screen._worker:
            screen._worker.stop(wait=True)


def test_demo_menu_has_expected_entries(qtbot, qt_theme_applied):
    """Menu wiring — every demo is a QAction under &Demos, including the
    real-dataset end-to-end option."""
    win = _new_mainwindow(qtbot, qt_theme_applied)
    demos_menu = None
    for act in win.menuBar().actions():
        if act.text().replace("&", "") == "Demos":
            demos_menu = act.menu()
            break
    assert demos_menu is not None, "no &Demos menu found"
    actions = [a for a in demos_menu.actions() if not a.isSeparator()]
    labels = {a.text() for a in actions}
    for expected in ("Mask demo…", "Measure demo…", "Crop demo…",
                      "Classify demo…", "Timelapse demo…",
                      "Sequencing demo…"):
        assert expected in labels
    # The real-dataset E2E option should be present
    assert any("End-to-end" in lbl and "Annotate" in lbl for lbl in labels)
