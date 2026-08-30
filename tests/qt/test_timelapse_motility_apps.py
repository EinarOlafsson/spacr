"""Timelapse and the Motility Assay as modules their hosts offer.

Both were checkboxes on Mask generation once, then modules of their own with
a tile each, and are now folded back into the screens they belong to without
losing anything on the way: Timelapse is a switch on the Mask masthead that
reveals its tracking settings categories, and the Motility Assay is a button
on the Measure masthead that opens its own screen. Neither has a registry row
any more, which is what folding one ends in.

The tests below pin what survived that:

  * both are still fully wired underneath — title, intro, icon, settings
    group, pre-flight rules and a pipeline entry point each — and both still
    run from ``spacr-run``;
  * neither has a tile or a sidebar row, and the host that carries each one
    names it;
  * Mask surfaces neither module's settings until its switch is pressed;
  * and nothing that used to work through the *pipeline* stopped working: an
    archived mask settings CSV with ``timelapse=True`` still drives
    ``preprocess_generate_masks`` exactly as before.
"""
from __future__ import annotations

import os

import pytest

import spacr.settings as S


NEW_APPS = ("timelapse", "motility")


# ---------------------------------------------------------------------------
# home screen
# ---------------------------------------------------------------------------

#: The host screen each folded module is reached from.
HOSTS = {"timelapse": "mask", "motility": "measure"}


@pytest.mark.parametrize("key", NEW_APPS)
def test_the_module_has_no_tile_and_its_host_offers_it(key):
    """Both were registry rows until they folded into the screens they suit.

    This asserted the row, its section and its stage. What replaces it is
    the same question asked of the fold: the tile is gone -- a tile that
    opens what a button on the host already opens is a second front door
    -- and the host that took it names it, so there is exactly one way in
    rather than none.
    """
    from spacr.qt.app import APPS
    from spacr.qt.screens import mask, measure

    assert not any(row[0] == key for row in APPS), (
        f"{key!r} still has a tile; the fold is not finished")
    folded = {"mask": mask.FOLDED_APPS, "measure": measure.FOLDED_APPS}
    assert key in folded[HOSTS[key]], (
        f"{key!r} is on no host, so nothing in the GUI can open it")


@pytest.mark.parametrize("key", NEW_APPS)
def test_the_button_still_says_what_the_tile_said(key):
    """The name, the sentence and the maturity colour survive the drop.

    ``FoldStrip`` reads all three out of the registry, which answers
    nothing once the row is gone -- an empty tooltip and a stable-blue
    hover for a module that is neither.
    """
    from spacr.qt.screens.map_barcodes import fold_description
    from spacr.qt.theme import STAGE_HOVER

    name, description, stage = fold_description(key)

    assert name and description
    assert stage in STAGE_HOVER
    for word in ("legacy", "deprecated", "experimental"):
        assert word not in f"{name} {description}".lower(), (
            f"{key} describes itself as {word!r}")


@pytest.mark.parametrize("key", NEW_APPS)
def test_new_module_has_a_title_and_an_intro(key):
    from spacr.qt.screens.app_screen import APP_INTROS, APP_TITLES
    assert key in APP_TITLES and APP_TITLES[key].strip()
    intro = APP_INTROS.get(key, "")
    # Same register + length as the existing blurbs: one informative sentence.
    assert len(intro.split()) >= 10, f"{key} intro is too thin: {intro!r}"
    assert intro.endswith(".")


@pytest.mark.parametrize("key", NEW_APPS)
def test_new_module_icon_resolves_to_a_real_resource_file(key):
    """A file on disk backs the tile, wherever the name comes from.

    This used to *require* an ``_ICON_OVERRIDES`` entry, because both
    modules borrowed somebody else's picture (timelapse→run.png,
    motility→recruitment.png). The user has since chosen artwork for
    both, and it is installed as ``<key>.png`` — which ``app_icon``
    finds with no override at all. Demanding an override entry would now
    mean demanding the borrowing back, so what is asserted is the thing
    that was always the point: the resolved filename exists."""
    from spacr.qt import app as qt_app
    here = os.path.dirname(os.path.abspath(qt_app.__file__))
    filename = qt_app._ICON_OVERRIDES.get(key, f"{key}.png")
    path = os.path.normpath(
        os.path.join(here, "..", "resources", "icons", filename))
    assert os.path.isfile(path), f"missing icon file: {path}"


@pytest.mark.parametrize("key", NEW_APPS)
def test_icon_provider_returns_an_icon(qtbot, qt_theme_applied, key):
    from PySide6.QtGui import QIcon
    from spacr.qt.app import _icon_for_app
    icon = _icon_for_app(key)
    assert isinstance(icon, QIcon)
    assert not icon.isNull(), f"{key} icon is null — the PNG failed to load"


def test_neither_module_is_on_the_sidebar_or_the_home_page(
        qtbot, qt_theme_applied):
    """Both are reached from a host now, and a second door is the thing
    the fold removed. Both surfaces are built from the registry, so a row
    on either would mean the drop had not really happened.
    """
    from PySide6.QtWidgets import QPushButton
    from spacr.qt.app import Sidebar, make_home_page
    from spacr.qt.widgets.home import AppTile

    bar = Sidebar()
    qtbot.addWidget(bar)
    labels = {b.accessibleName() for b in bar.findChildren(QPushButton)}
    assert "Timelapse" not in labels
    assert "Motility Assay" not in labels
    # The hosts are still there, or the modules are reachable from nothing.
    assert "Mask Generation" in labels or "Mask" in labels
    assert "Measure" in labels

    page = make_home_page()  # the page MainWindow ships
    qtbot.addWidget(page)
    tiles = {t.text_label for t in page.findChildren(AppTile)}
    assert tiles, "Home page rendered no tiles"
    assert not ({"Timelapse", "Motility Assay"} & tiles)


# ---------------------------------------------------------------------------
# pipeline wiring
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key,expected", [
    ("timelapse", "preprocess_generate_masks_timelapse"),
    ("motility", "automated_motility_assay"),
])
def test_resolve_pipeline_entry_returns_the_right_callable(key, expected):
    from spacr.qt.bridge import resolve_pipeline_entry
    entry = resolve_pipeline_entry(key)
    assert callable(entry), f"no pipeline entry for {key}"
    # log_call keeps __name__ (functools.wraps); fall back to __wrapped__.
    name = getattr(entry, "__name__", "")
    assert name == expected, f"{key} -> {name!r}, expected {expected!r}"


def test_timelapse_entry_forces_the_timelapse_flag(monkeypatch):
    """The module *is* timelapse; a False in the incoming dict is overridden."""
    import spacr.core as core
    seen = {}
    monkeypatch.setattr(core, "preprocess_generate_masks",
                        lambda s: seen.update(s))
    core.preprocess_generate_masks_timelapse({"src": "/tmp/x", "timelapse": False})
    assert seen["timelapse"] is True
    # and it still carries the whole mask settings surface
    for key in ("cell_channel", "batch_size", "timelapse_mode",
                "timelapse_objects", "timelapse_memory"):
        assert key in seen


def test_timelapse_override_is_announced_not_silent(monkeypatch, capsys):
    import spacr.core as core
    monkeypatch.setattr(core, "preprocess_generate_masks", lambda s: None)
    core.preprocess_generate_masks_timelapse({"src": "/tmp/x", "timelapse": False})
    assert "forcing it to True" in capsys.readouterr().out


def test_timelapse_entry_accepts_an_empty_dict(monkeypatch):
    import spacr.core as core
    monkeypatch.setattr(core, "preprocess_generate_masks", lambda s: s)
    out = core.preprocess_generate_masks_timelapse({})
    assert out["timelapse"] is True


def test_the_hosts_offer_both_modules_from_their_mastheads(
        qtbot, qt_theme_applied):
    """The one way in, driven the way a user drives it.

    Mask's is a switch that mounts the tracking categories on the form
    already on screen; Measure's is a button that opens the assay's own
    settings screen. This walked to a sidebar row until both rows were
    dropped, and the row is exactly what must not be what is checked.
    """
    from spacr.qt.app import MainWindow
    from spacr.qt.screens import mask as mask_folds
    from spacr.qt.screens import measure as measure_folds
    from spacr.qt.screens.app_screen import AppScreen

    win = MainWindow()
    qtbot.addWidget(win)

    win._on_nav_selected("mask")
    mask_screen = win._screens["mask"]
    assert mask_folds.install_folds(mask_screen) is not None
    folds = mask_folds.fold_set(mask_screen)
    assert folds is not None and "timelapse" in folds.order
    folds.strip.button_for("timelapse").setChecked(True)
    assert mask_screen._settings_model.collect()["timelapse"] is True

    win._on_nav_selected("measure")
    measure_screen = win._screens["measure"]
    strip = measure_folds.install_folds(measure_screen)
    assert strip is not None
    opener = next(o for o in measure_screen._fold_openers
                    if o.key == "motility")
    opened = opener.open()
    assert opened is not None
    assay = (opened if isinstance(opened, AppScreen)
               else next(iter(opened.findChildren(AppScreen)), None))
    assert assay is not None and assay.app_key == "motility"


def test_spacr_qt_timelapse_still_opens_the_timelapse_module(
        qtbot, qt_theme_applied):
    """``spacr-qt <app>`` is in shell histories and scripts.

    The key it names is a switch on Mask Generation now rather than a
    screen, so asking for it has to open the host WITH THE SWITCH ON --
    mask generation with tracking off is not the module that was asked
    for, and an orphan Timelapse page is not a module at all.
    """
    from spacr.qt.app import MainWindow
    from spacr.qt.screens import mask as mask_folds

    win = MainWindow(initial_app="timelapse")
    qtbot.addWidget(win)
    win.show()
    qt_theme_applied.processEvents()

    assert "timelapse" not in win._screens
    screen = win._screens["mask"]
    assert win._stack.currentWidget() is screen
    folds = mask_folds.fold_set(screen)
    assert folds is not None and folds.is_active("timelapse")
    assert screen._settings_model.collect()["timelapse"] is True


def test_opening_a_module_that_still_has_a_screen_is_unchanged(
        qtbot, qt_theme_applied):
    """The resolution must not move anything that was never folded."""
    from spacr.qt.app import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)

    assert win.open_module("measure") == "measure"
    assert win._stack.currentWidget() is win._screens["measure"]


def test_a_run_saved_before_the_fold_reopens_on_its_host(
        qtbot, qt_theme_applied):
    """Run History's "load this run's settings" must not build an orphan.

    Every run journal ever written names the module key that ran, and
    ``timelapse`` is in thousands of them. Navigating to a key with no row
    still builds a screen -- one with no sidebar entry, no tile and no way
    back to it -- so the key is resolved through the succession table
    first, and the gate the seed carries moves the switch the way an
    imported CSV does.
    """
    from spacr.qt.app import MainWindow
    from spacr.qt.chaining import screen_for_module
    from spacr.qt.screens import mask as mask_folds

    assert screen_for_module("timelapse") == "mask"
    assert screen_for_module("mask") == "mask"

    win = MainWindow()
    qtbot.addWidget(win)
    win.show()
    qt_theme_applied.processEvents()

    # Exactly what RunHistoryScreen.settings_requested emits for a run
    # recorded under the old key.
    win._on_train_requested("timelapse", {"src": "/data/p9",
                                            "timelapse_mode": "btrack",
                                            "timelapse": True})
    qt_theme_applied.processEvents()

    assert "timelapse" not in win._screens, "an orphan screen was built"
    screen = win._screens["mask"]
    assert win._stack.currentWidget() is screen
    folds = mask_folds.fold_set(screen)
    assert folds is not None and folds.is_active("timelapse")
    collected = screen._settings_model.collect()
    assert collected["src"] == "/data/p9"
    assert collected["timelapse_mode"] == "btrack"
    assert collected["timelapse"] is True


def test_the_timelapse_demo_lands_on_mask_with_tracking_switched_on(
        qtbot, qt_theme_applied, tmp_path):
    """The demo writes timelapse=True, and the switch is that key's control.

    It targeted the Timelapse module because Mask had no widget for the
    flag and it would have been silently dropped. It still has no widget
    -- the masthead switch is the control -- so what has to be true is
    that applying the demo's settings MOVES the switch.
    """
    from spacr.qt.app import MainWindow
    from spacr.qt import synthetic as syn
    from spacr.qt.screens import mask as mask_folds

    win = MainWindow()
    qtbot.addWidget(win)
    target_app, gen_name = win.DEMO_TARGETS["timelapse"]

    assert target_app == "mask"
    assert hasattr(syn, gen_name)

    win._on_nav_selected("mask")
    screen = win._screens["mask"]
    assert mask_folds.install_folds(screen) is not None
    screen.apply_settings_dict({"timelapse": "True",
                                 "timelapse_objects": ["cell"]})

    assert mask_folds.fold_set(screen).is_active("timelapse")
    assert screen._settings_model.collect()["timelapse"] is True


# ---------------------------------------------------------------------------
# settings groups
# ---------------------------------------------------------------------------

def test_timelapse_settings_group_exists_and_forces_the_flag():
    out = S.get_timelapse_settings({})
    assert out["timelapse"] is True
    # Everything spacr.object reads inside the `if timelapse:` branch.
    for key in ("timelapse_displacement", "timelapse_frame_limits",
                "timelapse_memory", "timelapse_remove_transient",
                "timelapse_mode", "timelapse_objects", "fps"):
        assert key in out, f"timelapse settings missing {key}"


def test_timelapse_settings_group_preserves_caller_values():
    out = S.get_timelapse_settings({"src": "/data/plate", "timelapse_memory": 9})
    assert out["src"] == "/data/plate"
    assert out["timelapse_memory"] == 9


def test_timelapse_settings_group_is_none_safe_and_unshared():
    a = S.get_timelapse_settings()
    b = S.get_timelapse_settings()
    a["__probe__"] = 1
    assert "__probe__" not in b


def test_timelapse_objects_uses_the_standard_list_editor(qtbot):
    from spacr.qt.screens.settings_model import SettingsWidgets, _ListEditor
    model = SettingsWidgets("timelapse")
    model.build_sections()
    widget = model._widgets["timelapse_objects"]
    assert isinstance(widget, _ListEditor)
    widget.set_value(["cell", "nucleus"])
    assert model.collect()["timelapse_objects"] == ["cell", "nucleus"]


def test_motility_settings_group_carries_its_own_source_folder():
    """It used to inherit `src` from the mask dict it was merged into."""
    out = S.get_automated_motility_assay_default_settings({})
    assert "src" in out
    assert out["db_table_name"] == "timelapse_object_measurements"


@pytest.mark.parametrize("key", NEW_APPS)
def test_settings_screen_builds_offscreen(qtbot, qt_theme_applied, key):
    from spacr.qt.screens.app_screen import AppScreen
    screen = AppScreen(key)
    qtbot.addWidget(screen)
    widgets = screen._settings_model._widgets
    assert widgets, f"{key} built no settings widgets"
    collected = screen._settings_model.collect()
    assert "src" in collected, f"{key} has no src setting"


def test_timelapse_screen_groups_axes_and_tracking(qtbot, qt_theme_applied):
    from spacr.qt.screens.settings_model import SettingsWidgets
    sections = dict(_section_map(SettingsWidgets("timelapse")))
    assert "Acquisition & Axes" in sections
    assert "Tracking Setup" in sections
    assert len(sections["Acquisition & Axes"]) >= 5


def test_motility_screen_shows_the_reorganized_categories(
    qtbot, qt_theme_applied,
):
    from spacr.qt.screens.settings_model import SettingsWidgets
    sections = dict(_section_map(SettingsWidgets("motility")))
    assert "Objects & Channels" in sections
    assert "Motion Filtering" in sections
    assert "Infection Classification" in sections
    assert "Motility Plots & QC" in sections
    assert "Other" not in sections, "motility keys spilled into the Other tab"


def _section_map(model):
    """[(section_title, [row_labels])] for a built SettingsWidgets."""
    return [(name, [label for label, _w in rows])
            for name, rows in model.build_sections()]


# ---------------------------------------------------------------------------
# removed from Mask
# ---------------------------------------------------------------------------

def test_mask_screen_has_no_timelapse_or_motility_categories(qtbot, qt_theme_applied):
    from spacr.qt.screens.settings_model import SettingsWidgets
    sections = dict(_section_map(SettingsWidgets("mask")))
    for gone in ("Timelapse", "Motility (beta)", "Motility Advanced (beta)"):
        assert gone not in sections, f"Mask still shows the {gone!r} tab"


def test_mask_categories_are_hidden_through_the_existing_mechanism():
    from spacr.qt.screens.settings_model import _APP_HIDDEN_CATEGORIES
    hidden = _APP_HIDDEN_CATEGORIES.get("mask", set())
    assert {"Timelapse", "Motility (beta)",
            "Motility Advanced (beta)"} <= hidden


def test_mask_defaults_expose_no_timelapse_or_motility_keys():
    """Not even via the trailing 'Other' bucket."""
    from spacr.qt.screens.settings_model import (
        resolve_default_settings, timelapse_and_motility_keys,
    )
    leaked = sorted(set(resolve_default_settings("mask"))
                    & timelapse_and_motility_keys())
    assert not leaked, f"Mask still edits {leaked}"


def test_mask_screen_still_offers_the_rest_of_its_settings(qtbot, qt_theme_applied):
    """Guard against the removal taking the whole panel with it."""
    from spacr.qt.screens.app_screen import AppScreen
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    collected = screen._settings_model.collect()
    for key in ("src", "cell_channel", "nucleus_channel", "batch_size",
                "magnification"):
        assert key in collected


def test_importing_a_legacy_mask_csv_reports_what_it_did_with_the_flags(
        qtbot, qt_theme_applied):
    """Non-modal console note, so the import is never silent (and never hangs).

    It said both flags had been IGNORED and sent the reader to a sidebar
    row for each. Neither half is true now: tracking is a switch on this
    form and the note has to report the switch moving, and the assay is a
    module Measure opens rather than a row anywhere.
    """
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.screens import mask as mask_folds

    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    assert mask_folds.install_folds(screen) is not None
    loaded = {"timelapse": "True", "motility_analysis": True}
    screen.apply_settings_dict(loaded)
    screen._warn_about_moved_settings(loaded)
    text = _console_text(screen)

    assert "switched Timelapse on" in text
    assert "motility_analysis=True was ignored" in text
    assert "Measure" in text
    assert "sidebar" not in text


def test_moved_settings_note_is_quiet_for_a_normal_csv(qtbot, qt_theme_applied):
    from spacr.qt.screens.app_screen import AppScreen
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    screen._warn_about_moved_settings({"timelapse": False, "src": "/data"})
    assert "Timelapse" not in _console_text(screen)


def _console_text(screen) -> str:
    from PySide6.QtWidgets import QLabel, QPlainTextEdit, QTextEdit
    parts = []
    for cls in (QPlainTextEdit, QTextEdit):
        parts += [w.toPlainText() for w in screen._console.findChildren(cls)]
    parts += [w.text() for w in screen._console.findChildren(QLabel)]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# nothing that used to work through the pipeline stopped working
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_plate(tmp_path):
    """A plate of CellVoyager-named raw tifs: 4 fields x 3 channels."""
    plate = tmp_path / "plate1"
    plate.mkdir()
    for field in range(1, 5):
        for chan in range(1, 4):
            (plate / f"plate1_A01_T0001F{field:03d}L01A01Z01C{chan:02d}.tif").write_bytes(b"")
    return plate


@pytest.fixture
def legacy_mask_csv(tmp_path, raw_plate):
    """A pre-split mask settings CSV: timelapse on, motility_analysis on."""
    csv = tmp_path / "gen_mask_settings.csv"
    rows = [
        ("src", str(raw_plate)),
        ("cell_channel", "0"),
        ("nucleus_channel", "1"),
        ("pathogen_channel", "2"),
        ("channels", "[0, 1, 2]"),
        ("magnification", "20"),
        ("timelapse", "True"),
        ("timelapse_mode", "trackpy"),
        ("timelapse_memory", "5"),
        ("timelapse_objects", "['cell']"),
        ("timelapse_displacement", "40"),
        ("timelapse_remove_transient", "True"),
        ("timelapse_frame_limits", "[0, 8]"),
        ("fps", "4"),
        ("motility_analysis", "True"),
    ]
    csv.write_text("Key,Value\n" + "\n".join(f'{k},"{v}"' for k, v in rows) + "\n")
    return csv


def test_legacy_csv_still_parses_with_every_timelapse_key(legacy_mask_csv):
    from spacr.utils import load_settings
    loaded = load_settings(str(legacy_mask_csv),
                           setting_key="Key", setting_value="Value")
    assert loaded["timelapse"] is True
    assert loaded["motility_analysis"] is True
    assert loaded["timelapse_memory"] == 5


def test_mask_pipeline_defaults_still_carry_the_timelapse_flags():
    """spacr.object reads settings['timelapse'] on EVERY mask run."""
    out = S.set_default_settings_preprocess_generate_masks({})
    assert out["timelapse"] is False
    assert out["motility_analysis"] is False
    for key in ("fps", "timelapse_mode", "timelapse_objects",
                "timelapse_memory", "timelapse_displacement",
                "timelapse_frame_limits", "timelapse_remove_transient"):
        assert key in out, f"mask pipeline defaults lost {key}"


def test_mask_pipeline_honours_a_legacy_timelapse_true(legacy_mask_csv):
    from spacr.utils import load_settings
    loaded = load_settings(str(legacy_mask_csv),
                           setting_key="Key", setting_value="Value")
    out = S.set_default_settings_preprocess_generate_masks(dict(loaded))
    assert out["timelapse"] is True, "a legacy CSV's timelapse=True was dropped"
    assert out["motility_analysis"] is True
    assert out["timelapse_memory"] == 5


def test_mask_pipeline_runs_end_to_start_on_a_legacy_timelapse_csv(
        legacy_mask_csv, raw_plate, capsys):
    """dry_run drives the full settings->validate path with no compute."""
    import spacr.core as core
    from spacr.utils import load_settings
    settings = load_settings(str(legacy_mask_csv),
                             setting_key="Key", setting_value="Value")
    settings["dry_run"] = True
    problems = core.preprocess_generate_masks(settings)   # must not raise
    fatal = [p for p in problems
             if p.severity == "error" and p.setting in (
                 "timelapse", "motility_analysis", "timelapse_mode",
                 "timelapse_objects")]
    assert not fatal, f"legacy timelapse settings now fail preflight: {fatal}"
    assert "spacr.core.preprocess_generate_masks" in capsys.readouterr().out


def test_object_module_reads_the_timelapse_flag_defensively():
    """A hand-built settings dict must not KeyError before segmentation."""
    import inspect
    import spacr.object as obj
    src = inspect.getsource(obj)
    assert "settings['timelapse']" not in src, (
        "spacr.object still hard-indexes settings['timelapse']; the Mask GUI "
        "no longer supplies that key, so it must be read with .get()")


def test_inline_motility_hook_is_still_gated_on_both_flags():
    import inspect
    import spacr.object as obj
    src = inspect.getsource(obj)
    assert src.count(
        'if timelapse and settings.get("motility_analysis", False):') == 2


# ---------------------------------------------------------------------------
# tooltip quality gate for everything the new modules expose
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", NEW_APPS)
def test_every_setting_the_new_modules_expose_has_a_tooltip(key):
    from spacr.qt.screens.settings_model import resolve_default_settings
    missing = sorted(k for k in resolve_default_settings(key)
                     if k not in S.tooltips)
    # `pipeline_style`, `batch_fields`, `keep_npz` etc. are shared with mask
    # and already exempt there; anything else is a real gap.
    already_exempt = {k for k in missing if k not in S.expected_types}
    assert not (set(missing) - already_exempt), (
        f"{key} exposes settings with no tooltip: {sorted(set(missing) - already_exempt)}")


@pytest.mark.parametrize("key", NEW_APPS)
def test_new_module_tooltips_meet_the_house_bar(key):
    """Same floors as tests/test_settings_tooltip_quality.py."""
    from spacr.qt.screens.settings_model import resolve_default_settings
    from tests.test_settings_tooltip_quality import TYPE_PREFIX

    offenders = []
    for name in resolve_default_settings(key):
        text = S.tooltips.get(name)
        if not text:
            continue
        m = TYPE_PREFIX.match(text.strip())
        if m is None:
            offenders.append((name, "no (type) prefix"))
            continue
        body = m.group("body").strip()
        if len(body.split()) < 15:
            offenders.append((name, "under 15 words"))
        if "\n" in body or "**" in body or "`" in body:
            offenders.append((name, "not single-line plain text"))
    assert not offenders, f"{key}: {offenders}"


def test_settings_lists_and_categories_stay_in_sync():
    """`timelapse_settings` is the single source of the Timelapse tab."""
    assert S.categories["Timelapse"] is S.timelapse_settings
    assert S.categories["Motility (beta)"] is S.motility_settings
    # `timelapse` itself must stay OUT of the category it controls: the Tk GUI
    # reveals that category only once the box is ticked.
    assert "timelapse" not in S.timelapse_settings
    assert "timelapse" in S.categories["General"]
