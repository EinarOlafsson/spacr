"""Every app the GUI offers must be reachable without the GUI, or say why not.

spaCR keeps four registries of "what can be run":

  * ``spacr.qt.app.APPS``                       — the buttons
  * ``spacr.qt.bridge.resolve_pipeline_entry``  — what a button runs
  * ``spacr.cli.MODULES``                       — what ``spacr-run`` can run
  * ``spacr.validate.APP_FUNCTIONS``            — what pre-flight has rules for

Adding an app touches the first two, because that is what makes the button
appear and work. Nothing forces the other two, and nothing noticed when they
were skipped:

  * ``invasion`` shipped with a Qt button, a settings panel, a settings
    category and ``spacr.submodules.analyze_invasion`` behind it — and no
    ``spacr-run invasion`` and no pre-flight entry, for its entire life.
  * ``replication`` was wired into Qt in ``7c65784`` and reached
    ``validate.APP_FUNCTIONS`` there, but not ``cli.MODULES``.
  * ``foreign`` had a button and a pre-flight entry but no CLI module.
  * ``timelapse``, ``motility`` and ``activation`` had buttons and CLI modules
    but no pre-flight entry, so ``validate_settings(settings, 'timelapse')``
    answered *"unknown app; only the generic checks were run"* and skipped the
    segmentation-channel rules that would have caught an empty channel set.

Every one of those is the same shape: an app that works on a workstation and
does not exist on a cluster, discovered by a user with a 40-plate job to run.
So the registries are asserted against each other here rather than trusted, in
both directions — an app that is genuinely GUI-only is not a failure, but it
has to *say so*, in ``cli.INTERACTIVE_ONLY``, with a sentence telling the user
what to do instead.
"""
from __future__ import annotations

import os
import sqlite3

import pandas as pd
import pytest

from spacr import cli
from spacr.qt.app import APPS
from spacr.validate import APP_FUNCTIONS

APP_KEYS = [key for key, _name, _desc, _section in APPS]


# Modules that are runnable headless but have no button, which is a legitimate
# direction to be missing in — the reason is recorded here so that "not in
# APPS" cannot quietly become the excuse for a fourth unwired app.
HEADLESS_ONLY = {
    "train_only": "the training stage of `classify` on an existing dataset "
                  "folder; the Classify button runs the whole pipeline",
    "cellpose_all": "benchmarks every Cellpose model on one folder; Model Zoo "
                    "is the interactive version of the same question",
    "endodyogeny": "the legacy area-derived size proxy remains headless; the "
                   "Replication button runs the parasite-count assay",
    "simulation": "the pooled-screen simulator, which has no GUI screen at all",
}


def _entry_name(fn):
    """The real function behind a bridge entry (they are wrapped by log_call)."""
    inner = getattr(fn, "__wrapped__", fn)
    return getattr(inner, "__name__", None) or getattr(fn, "__name__", "")


def _qt_entry_name(app_key):
    from spacr.qt.bridge import resolve_pipeline_entry
    fn = resolve_pipeline_entry(app_key)
    return _entry_name(fn) if fn is not None else ""


# ---------------------------------------------------------------------------
# The point of this file
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("app_key", APP_KEYS)
def test_every_app_has_a_cli_module_or_is_declared_gui_only(app_key):
    """A button with no ``spacr-run`` module cannot be used on a cluster."""
    if app_key in cli.INTERACTIVE_ONLY:
        return
    assert app_key in cli.MODULES, (
        f"'{app_key}' is in spacr.qt.app.APPS but has no spacr.cli.MODULES "
        f"entry, so `spacr-run {app_key}` answers 'unknown module'. Add the "
        f"Module, or add it to cli.INTERACTIVE_ONLY with a sentence saying "
        f"what to do instead.")


@pytest.mark.parametrize("app_key", APP_KEYS)
def test_every_app_has_a_validate_entry_or_is_declared_gui_only(app_key):
    """Without an entry, pre-flight calls the app unknown and skips its rules."""
    if app_key in cli.INTERACTIVE_ONLY:
        return
    assert app_key in APP_FUNCTIONS, (
        f"'{app_key}' is in spacr.qt.app.APPS but has no "
        f"spacr.validate.APP_FUNCTIONS entry, so validate_settings warns "
        f"'unknown app' and runs the generic checks only.")


@pytest.mark.parametrize("app_key", sorted(cli.INTERACTIVE_ONLY))
def test_every_gui_only_app_gives_a_reason_and_an_alternative(app_key):
    """"GUI-only" is an acceptable answer; a blank one is not.

    ``spacr-run <key>`` prints this text instead of "unknown module", so it is
    the only thing standing between a user and the conclusion that they typed
    the name wrong.
    """
    reason = cli.INTERACTIVE_ONLY[app_key]
    assert isinstance(reason, str) and len(reason) >= 40, (
        f"{app_key}: the GUI-only reason is too short to be one: {reason!r}")
    assert reason.strip().rstrip(".") != app_key
    lowered = reason.lower()
    assert any(marker in lowered for marker in
               ("spacr-qt", "gui", "headless", "instead", "call ", "run ")), (
        f"{app_key}: the reason says the app is interactive but not what to do "
        f"instead: {reason!r}")


def test_gui_only_and_runnable_are_disjoint():
    """An app cannot be both; the error message picks one branch."""
    assert not (set(cli.INTERACTIVE_ONLY) & set(cli.MODULES))


def test_the_gui_only_list_holds_no_apps_that_no_longer_exist():
    """A stale entry hides a genuinely unknown module behind a helpful lie."""
    ghosts = sorted(set(cli.INTERACTIVE_ONLY) - set(APP_KEYS))
    assert not ghosts, (
        f"cli.INTERACTIVE_ONLY names apps that are not in spacr.qt.app.APPS: "
        f"{ghosts}")


def test_every_cli_module_is_an_app_or_a_declared_headless_only_one():
    """The other direction: a module nobody can reach from the GUI."""
    orphans = sorted(set(cli.MODULES) - set(APP_KEYS) - set(HEADLESS_ONLY))
    assert not orphans, (
        f"spacr.cli.MODULES has modules with no APPS entry and no reason "
        f"recorded in this test's HEADLESS_ONLY: {orphans}")
    stale = sorted(set(HEADLESS_ONLY) - set(cli.MODULES))
    assert not stale, f"HEADLESS_ONLY names modules that no longer exist: {stale}"


def test_every_validate_entry_is_an_app_or_a_headless_only_module():
    orphans = sorted(set(APP_FUNCTIONS) - set(APP_KEYS) - set(HEADLESS_ONLY))
    assert not orphans, (
        f"spacr.validate.APP_FUNCTIONS describes apps that exist nowhere else: "
        f"{orphans}")


# ---------------------------------------------------------------------------
# and the entries have to name the same function
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("app_key", APP_KEYS)
def test_the_registries_name_the_same_function(app_key):
    """Validating against one function and running another is not validation."""
    if app_key in cli.INTERACTIVE_ONLY:
        return
    names = {}
    qt_name = _qt_entry_name(app_key)
    if qt_name:
        names["qt bridge"] = qt_name
    if app_key in cli.MODULES:
        names["spacr-run"] = cli.MODULES[app_key].func_name
    if app_key in APP_FUNCTIONS:
        names["validate"] = APP_FUNCTIONS[app_key].rsplit(".", 1)[-1]
    assert len(set(names.values())) == 1, (
        f"{app_key}: the registries disagree about what runs — {names}")


@pytest.mark.parametrize("app_key", ["invasion", "replication"])
def test_the_two_toxo_assays_resolve_end_to_end(app_key):
    """The gap this file was written for, closed: name in, callable out."""
    module = cli.resolve_module(app_key)
    assert module is not None
    assert module.module_name == "spacr.submodules"
    assert module.validate_key == app_key
    from spacr import settings as spacr_settings
    assert callable(getattr(spacr_settings, module.defaults))


# ---------------------------------------------------------------------------
# …and the wiring does something when it is used
# ---------------------------------------------------------------------------

def _real_plate(root):
    """A plate whose measurements.db is written by spaCR's own writer.

    :func:`spacr.utils._merge_and_save_to_database` is what ``measure_crop``
    appends every object table with, so the schema, the derived
    plateID/rowID/columnID/fieldID columns and the prcf key are the ones a real
    run leaves behind — not a hand-built table that happens to satisfy the
    check under test.
    """
    from spacr.utils import _merge_and_save_to_database

    root = str(root)
    os.makedirs(os.path.join(root, "measurements"), exist_ok=True)
    for index, stem in enumerate(("plate1_A01_1", "plate1_A01_2")):
        pathogen_morph = pd.DataFrame(
            {"label": [1, 2],
             # Object-table parent ids are labels within this field; the
             # writer adds the field/plate identity separately as ``prcf``.
             "cell_id": [1, 2],
             "pathogen_area": [500.0 + index, 1100.0 + index]})
        pathogen_intensity = pd.DataFrame(
            {"label": [1, 2],
             "pathogen_channel_1_mean_intensity": [120.0, 4000.0]})
        _merge_and_save_to_database(pathogen_morph, pathogen_intensity,
                                    "pathogen", root, stem, "spacr_run")
    return os.path.join(root, "measurements", "measurements.db")


@pytest.mark.parametrize("app_key", ["invasion", "replication"])
def test_preflight_reports_a_missing_measurements_db(tmp_path, app_key):
    """Both assays open ``<src>/measurements/measurements.db``; say so first."""
    from spacr.validate import validate_settings

    plate = tmp_path / f"{app_key}_plate"
    plate.mkdir()
    problems = validate_settings({"src": str(plate)}, app_key)
    errors = [p for p in problems if p.is_error]
    assert any("measurements database not found" in p.message for p in errors), \
        [str(p) for p in problems]
    assert not any("unknown app" in p.message for p in problems)


@pytest.mark.parametrize("app_key", ["invasion", "replication"])
def test_preflight_passes_on_a_plate_measure_crop_really_wrote(tmp_path, app_key):
    plate = tmp_path / f"{app_key}_plate"
    plate.mkdir()
    db = _real_plate(plate)
    assert os.path.isfile(db)
    with sqlite3.connect(db) as connection:
        assert connection.execute(
            "SELECT count(*) FROM pathogen").fetchone()[0] == 4

    from spacr.validate import validate_settings
    module = cli.MODULES[app_key]
    settings = cli.resolve_settings(module, None, [f"src={plate}"])
    problems = validate_settings(settings, module.validate_key)
    assert [str(p) for p in problems if p.is_error] == []


def test_spacr_run_invasion_is_no_longer_an_unknown_module(capsys):
    """The measured symptom: `spacr-run invasion` used to say 'unknown module
    invasion. Did you mean activation?' and exit 2."""
    assert cli.main(["--describe", "invasion"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "spacr.submodules.analyze_invasion" in out
    assert "unknown module" not in out


def test_the_new_modules_are_listed(capsys):
    assert cli.main(["--list"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    for key in ("invasion", "replication", "foreign"):
        assert f"  {key}" in out, key


@pytest.mark.parametrize("spelling,expected", [
    ("invasion", "invasion"),
    ("analyze_invasion", "invasion"),
    ("Invasion-Assay", "invasion"),
    ("replication", "replication"),
    ("analyze_replication", "replication"),
    ("analyze_endodyogeny", "endodyogeny"),
    ("endodyogeny", "endodyogeny"),
    ("import_project", "foreign"),
])
def test_the_new_names_resolve_the_way_a_pasted_script_spells_them(spelling,
                                                                   expected):
    module = cli.resolve_module(spelling)
    assert module is not None and module.key == expected


def test_timelapse_now_gets_the_mask_pipelines_rules_not_the_generic_ones(tmp_path):
    """``preprocess_generate_masks_timelapse`` prints "At least one of
    cell_channel, nucleus_channel, pathogen_channel or organelle_channel must
    be defined" and returns, exactly as the mask pipeline does. Before
    ``timelapse`` was in APP_FUNCTIONS, pre-flight called it an unknown app and
    let that settings file through."""
    from spacr.validate import validate_settings

    src = tmp_path / "plate"
    src.mkdir()
    (src / "plate1_A01_T0001F001L01A01Z01C01.tif").write_bytes(b"")
    settings = {"src": str(src), "cell_channel": None, "nucleus_channel": None,
                "pathogen_channel": None, "organelle_channel": None}
    problems = validate_settings(settings, "timelapse")
    assert any(p.is_error and p.setting == "cell_channel" for p in problems), \
        [str(p) for p in problems]
    assert not any("unknown app" in p.message for p in problems)


# ---------------------------------------------------------------------------
# foreign: the third gap, and the two shared-namespace traps behind it
# ---------------------------------------------------------------------------

def test_foreign_resolves_the_defaults_its_own_module_owns():
    """``spacr.foreign`` keeps its defaults factory, not ``spacr.settings``.

    With no defaults resolved, every ``--set`` against a foreign key would be
    rejected as "a setting that does not exist" — an import you could describe
    but not configure.
    """
    module = cli.MODULES["foreign"]
    assert module.defaults is None
    assert module.defaults_entry == "spacr.foreign:default_settings"
    defaults = cli.module_defaults(module)
    assert {"images", "masks", "measurements", "preview_only"} <= set(defaults)
    assert "spacr.foreign.default_settings()" in module.defaults_label


def test_the_two_per_app_type_override_tables_agree():
    """``spacr.cli`` mirrors ``spacr.validate`` rather than importing it (the
    CLI must answer --list without loading anything). A value the validator
    accepts has to be a value ``--set`` can write, so the mirror is pinned."""
    from spacr.validate import _APP_TYPE_OVERRIDES as validate_overrides

    assert cli._APP_TYPE_OVERRIDES == validate_overrides


def test_a_key_two_pipelines_share_is_read_as_the_module_means_it():
    """``masks`` is bool for the mask pipeline and a path for a foreign import."""
    from spacr.settings import expected_types

    assert expected_types["masks"] is bool
    assert cli.coerce_value("masks", "/their/masks", None, expected_types,
                            "foreign") == "/their/masks"
    with pytest.raises(cli.SettingsError):
        cli.coerce_value("masks", "/their/masks", None, expected_types, "mask")


def test_the_foreign_key_list_validate_carries_matches_the_real_factory():
    """``spacr.validate`` cannot import ``spacr.foreign`` (it would pull
    ``spacr.convert`` into a module that promises to import only
    ``spacr.settings``), so it carries the key list. Pin it here."""
    from spacr.foreign import default_settings
    from spacr.validate import _APP_EXTRA_KEYS

    assert _APP_EXTRA_KEYS["foreign"] == frozenset(default_settings({}))


def test_foreign_preflight_does_not_invent_problems(tmp_path):
    """Every one of these was a real false positive on a correct settings file.

    ``src`` is not a foreign key at all; ``masks`` is declared bool in the
    shared ``expected_types`` (the mask pipeline's save switch) but is a path
    here; and ``measurements`` reads as a typo of ``measurement``.
    """
    from spacr.validate import validate_settings

    images = tmp_path / "theirs"
    masks = tmp_path / "theirs_masks"
    images.mkdir()
    masks.mkdir()
    table = tmp_path / "theirs.csv"
    table.write_text("object,area\n1,10\n")

    module = cli.MODULES["foreign"]
    settings = cli.resolve_settings(module, None, [
        f"images={images}", f"masks={masks}", f"measurements={table}",
        "preview_only=True"])
    problems = validate_settings(settings, module.validate_key)
    assert [str(p) for p in problems] == []


def test_foreign_preflight_reports_a_path_that_is_not_there(tmp_path):
    from spacr.validate import validate_settings

    images = tmp_path / "theirs"
    images.mkdir()
    settings = dict(cli.module_defaults(cli.MODULES["foreign"]))
    settings.update({"images": str(images),
                     "masks": str(tmp_path / "gone"),
                     "measurements": str(tmp_path / "gone.csv"),
                     "column_map": str(tmp_path / "map.json")})
    missing = {p.setting for p in validate_settings(settings, "foreign")
               if p.is_error and "does not exist" in p.message}
    assert missing == {"masks", "measurements"}


def test_the_missing_column_map_warning_only_fires_on_a_real_run(tmp_path):
    """A preview writes nothing, so an inferred mapping is exactly the point."""
    from spacr.validate import validate_settings

    def _warnings(**over):
        settings = dict(cli.module_defaults(cli.MODULES["foreign"]))
        settings.update(over)
        return [p.setting for p in validate_settings(settings, "foreign")
                if not p.is_error]

    assert "column_map" in _warnings()
    assert "column_map" not in _warnings(preview_only=True)


def test_a_defaults_helper_that_cannot_be_imported_does_not_break_describe():
    """--describe has to answer even against a half-installed environment."""
    broken = cli.Module(key="_broken", summary="", entry="spacr.foreign:import_project",
                        defaults=None, validate_key="",
                        defaults_entry="spacr_no_such_module:default_settings")
    assert cli.module_defaults(broken) == {}
    assert broken.defaults_label == "spacr_no_such_module.default_settings()"


def test_describe_names_whichever_defaults_helper_a_module_has():
    assert cli.MODULES["measure"].defaults_label == \
        "spacr.settings.get_measure_crop_settings()"
    assert cli.MODULES["foreign"].defaults_label == \
        "spacr.foreign.default_settings()"
    assert cli.MODULES["align"].defaults_label == ""


def test_foreign_preflight_says_what_is_missing(tmp_path):
    """import_project raises ConfigurationError for each of these; pre-flight
    has to get there first, and name the right key."""
    from spacr.validate import validate_settings

    module = cli.MODULES["foreign"]
    problems = validate_settings(cli.module_defaults(module),
                                 module.validate_key)
    named = {p.setting for p in problems if p.is_error}
    assert {"images", "masks", "measurements"} <= named
    assert "src" not in named


def test_replication_uses_counts_and_keeps_the_size_proxy_explicit():
    """The visible assay counts parasites; the legacy proxy stays named."""
    assert cli.MODULES["replication"].func_name == "analyze_replication"
    assert cli.MODULES["endodyogeny"].func_name == "analyze_endodyogeny"
    assert cli.resolve_module("analyze_replication").key == "replication"
    assert cli.resolve_module("analyze_endodyogeny").key == "endodyogeny"


@pytest.mark.parametrize(("key", "value"), [
    ("max_parasites_per_vacuole", 12),
    ("vacuole_link_factor", 0),
    ("vacuole_link_distance", -1),
    ("non_power_of_two_warn", 1.2),
])
def test_replication_preflight_rejects_invalid_scoring_settings(key, value):
    """Scientifically invalid knobs fail before database loading."""
    from spacr.settings import set_analyze_replication_defaults
    from spacr.validate import validate_settings

    settings = set_analyze_replication_defaults({})
    settings[key] = value
    problems = validate_settings(settings, "replication")
    assert any(p.setting == key and p.is_error for p in problems)
