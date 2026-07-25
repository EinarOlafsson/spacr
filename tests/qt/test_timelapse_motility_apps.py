"""Timelapse and Motility Assay as first-class home-screen modules.

Timelapse used to be a checkbox on Mask generation and the automated motility
assay a pair of "(beta)" tabs hidden behind ``motility_analysis``. Both are now
modules of their own — an ``APPS`` entry, a title, an intro, an icon, a
settings group and a pipeline entry point each — and Mask offers neither.

The tests below pin all three halves of that:

  * the two new modules are fully wired (home screen -> screen -> pipeline),
  * Mask no longer surfaces either group,
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

@pytest.mark.parametrize("key", NEW_APPS)
def test_new_module_is_on_the_home_screen(key):
    from spacr.qt.app import APPS
    entry = next((a for a in APPS if a[0] == key), None)
    assert entry is not None, f"{key!r} missing from APPS"
    app_key, name, desc, section = entry
    assert name and desc
    assert section == "Core", (
        f"{key!r} is a first-class workflow, not a Tool — got section {section!r}")


def test_timelapse_is_not_filed_as_legacy_or_beta():
    """The user's ask: timelapse is a first-class capability."""
    from spacr.qt.app import APPS
    _, name, desc, _ = next(a for a in APPS if a[0] == "timelapse")
    blob = f"{name} {desc}".lower()
    for word in ("legacy", "deprecated", "beta", "experimental"):
        assert word not in blob, f"timelapse entry calls itself {word!r}"


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
    from spacr.qt import app as qt_app
    assert key in qt_app._ICON_OVERRIDES, (
        f"{key} has no dedicated icon; add an _ICON_OVERRIDES entry")
    here = os.path.dirname(os.path.abspath(qt_app.__file__))
    path = os.path.join(here, "..", "resources", "icons",
                        qt_app._ICON_OVERRIDES[key])
    assert os.path.isfile(os.path.normpath(path)), f"missing icon file: {path}"


@pytest.mark.parametrize("key", NEW_APPS)
def test_icon_provider_returns_an_icon(qtbot, qt_theme_applied, key):
    from PySide6.QtGui import QIcon
    from spacr.qt.app import _icon_for_app
    icon = _icon_for_app(key)
    assert isinstance(icon, QIcon)
    assert not icon.isNull(), f"{key} icon is null — the PNG failed to load"


def test_sidebar_and_startup_page_render_the_new_modules(qtbot, qt_theme_applied):
    from PySide6.QtWidgets import QPushButton
    from spacr.qt.app import APPS, Sidebar, _icon_for_app
    from spacr.qt.screens.startup import StartupPage

    bar = Sidebar()
    qtbot.addWidget(bar)
    labels = {b.accessibleName() for b in bar.findChildren(QPushButton)}
    assert "Timelapse" in labels
    assert "Motility Assay" in labels

    page = StartupPage(APPS, _icon_for_app)
    qtbot.addWidget(page)
    from spacr.qt.widgets.tile import HTile
    assert page.findChildren(HTile), "startup page rendered no tiles"


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


def test_main_window_builds_both_new_screens(qtbot, qt_theme_applied):
    from spacr.qt.app import MainWindow
    from spacr.qt.screens.app_screen import AppScreen
    win = MainWindow()
    qtbot.addWidget(win)
    for key in NEW_APPS:
        win._on_nav_selected(key)
        screen = win._screens.get(key)
        assert isinstance(screen, AppScreen)
        assert screen.app_key == key


def test_timelapse_demo_targets_the_timelapse_module(qtbot, qt_theme_applied):
    """The demo writes timelapse=True; Mask has no widget for it any more."""
    from spacr.qt.app import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)
    target_app, gen_name = win.DEMO_TARGETS["timelapse"]
    assert target_app == "timelapse"
    from spacr.qt import synthetic as syn
    assert hasattr(syn, gen_name)


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


def test_timelapse_screen_shows_the_timelapse_category(qtbot, qt_theme_applied):
    from spacr.qt.screens.settings_model import SettingsWidgets
    sections = dict(_section_map(SettingsWidgets("timelapse")))
    assert "Timelapse" in sections
    assert len(sections["Timelapse"]) >= 5


def test_motility_screen_shows_the_motility_categories(qtbot, qt_theme_applied):
    from spacr.qt.screens.settings_model import SettingsWidgets
    sections = dict(_section_map(SettingsWidgets("motility")))
    assert "Motility (beta)" in sections
    assert "Motility Advanced (beta)" in sections
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


def test_importing_a_legacy_mask_csv_reports_the_moved_settings(
        qtbot, qt_theme_applied):
    """Non-modal console note, so the drop is never silent (and never hangs)."""
    from spacr.qt.screens.app_screen import AppScreen
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    screen._warn_about_moved_settings(
        {"timelapse": "True", "motility_analysis": True})
    text = screen._console.toPlainText() if hasattr(screen._console, "toPlainText") \
        else _console_text(screen)
    assert "Timelapse module" in text
    assert "Motility Assay module" in text


def test_moved_settings_note_is_quiet_for_a_normal_csv(qtbot, qt_theme_applied):
    from spacr.qt.screens.app_screen import AppScreen
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    screen._warn_about_moved_settings({"timelapse": False, "src": "/data"})
    assert "Timelapse module" not in _console_text(screen)


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
    from tests.test_settings_tooltip_quality import KNOWN_THIN, TYPE_PREFIX
    from spacr.qt.screens.settings_model import resolve_default_settings

    offenders = []
    for name in resolve_default_settings(key):
        text = S.tooltips.get(name)
        if not text or name in KNOWN_THIN:
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
