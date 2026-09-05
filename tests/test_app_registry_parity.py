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
    # An ALIAS now, not a Module of its own: "benchmark every Cellpose model
    # on one folder" became "Mask the whole folder" on the Make Masks
    # masthead, and `cellpose_all` was pointed at `cellpose_masks` so a script
    # or settings CSV that still names it keeps running. Kept here because
    # `spacr-run cellpose_all` still resolves and still has no tile.
    "cellpose_all": "an alias of `cellpose_masks` kept so older scripts run; "
                    "the interactive version is the Make Masks masthead's "
                    "\u201cMask the whole folder\u201d button",
    "endodyogeny": "the legacy area-derived size proxy remains headless; the "
                   "Replication button runs the parasite-count assay",
    "simulation": "the pooled-screen simulator, which has no GUI screen at all",
    # THE TWO CLASSIFY SCREENS WERE REMOVED, their entry points were not.
    # "Classify (CV)" and "Classify (ML)" were the originals kept beside the
    # merged screen; three entries for one job is three places to look, and
    # two of them are the same run with half the choices. Removed from the
    # app registry on 2026-08-23.
    #
    # They stay in `cli.MODULES` and `validate.APP_FUNCTIONS` on purpose: a
    # settings CSV written for either still runs, and a notebook importing
    # `deep_spacr` or `generate_ml_scores` still works. The merged screen
    # dispatches to exactly those two functions, so nothing is duplicated --
    # only the second and third front doors are gone.
    "classify": "the CV half of Classify, still runnable headless and from a "
                "settings CSV; the Classify screen is the GUI for it",
    "ml_analyze": "the ML half of Classify, same arrangement as `classify`",
    # THE SAME ARRANGEMENT AGAIN, and it went unrecorded when it happened:
    # Cellpose Masks lost its row to the merged Cellpose Workbench, whose
    # Apply tab it is. `spacr-run cellpose_masks` and a settings CSV written
    # for it both still work, and this ledger has been red about it ever
    # since -- not folded (no host offers `cellpose_masks` as a button), just
    # the second half of a page reached by its tab.
    "cellpose_masks": "the applying half of the Cellpose Workbench, reached "
                      "as that page's Apply tab; still runnable headless and "
                      "from a settings CSV written before the merge",
}


# Modules that were FOLDED into a host screen and then had their registry row
# dropped. "Not in APPS" is the point for these, not an omission: the tile went
# and the module did not. Each is still runnable headless, still validated, and
# still reachable in the GUI -- from the host named beside it -- so the reason
# recorded here is WHERE THE USER FINDS IT NOW.
#
# The rule this preserves is the one HEADLESS_ONLY protects from the other
# side: "missing from APPS" must always come with a reason somebody wrote
# down, or a genuinely unwired module hides behind the exception.
FOLDED = {
    "ops": "a button on the Align & Stitch masthead that opens the optical-"
           "pooled-screening form as a page beside the tile aligner -- OPS "
           "IS stitching, over a plate acquired in sequencing cycles, so it "
           "is reached from the module it belongs to rather than competing "
           "for a tile of its own (spacr.qt.screens.align)",
    "activation": "a button on the Classify masthead that opens the "
                  "activation-map workbench as a page beside the training "
                  "settings -- an activation map is a view of what a "
                  "trained classifier looked at, which is Classify's own "
                  "output (spacr.qt.screens.classify)",
    "illumination": "nine settings on the Measure panel rather than a "
                    "button, because the flat-field correction is applied "
                    "BEFORE any intensity feature is computed -- it has to "
                    "be settable on the measure run it changes, and a "
                    "separate screen could only ever be a second place to "
                    "set the same thing (spacr.qt.screens.measure)",
    "timelapse": "a switch on the Mask Generation masthead that reveals its "
                 "tracking settings categories and turns the pipeline's "
                 "`timelapse` gate on (spacr.qt.screens.mask)",
    "motility": "a button on the Measure masthead that opens the assay's own "
                "screen as a page beside the measure form "
                "(spacr.qt.screens.measure)",
    "agreement": "a button on the Annotate masthead that opens the kappa "
                 "table and the disagreement review, whole "
                 "(spacr.qt.screens.annotate)",
    "barcode_qc": "a button on the Map Barcodes masthead that opens the "
                  "QC's own settings form and Run button as a page beside "
                  "the mapping settings, so the mapping and its QC are one "
                  "visit (spacr.qt.screens.map_barcodes)",
    "classifier_evaluation": "a button on the Classify masthead that opens "
                             "the evaluation bundle browser -- held-out "
                             "predictions, calibration, per-plate metrics "
                             "and the leakage report -- as a page beside "
                             "the training settings "
                             "(spacr.qt.screens.classify)",
    "explain_cv": "a button on the Classify masthead that opens the "
                  "fidelity, permutation-importance and SHAP workbench as a "
                  "page beside the training settings "
                  "(spacr.qt.screens.classify)",
    "anndata_export": "a button on the Measure masthead that opens the "
                      "export's own settings form and Run button as a page "
                      "beside the measure settings "
                      "(spacr.qt.screens.measure)",
    "image_scatter": "a button on the Image UMAP masthead that opens the "
                     "scatter's own screen as a page, pointed at the "
                     "measurements database the UMAP screen is reading "
                     "(spacr.qt.screens.image_umap)",
    "pca": "a button on the Image UMAP masthead that opens the decomposition "
           "with its feature picker, scree plot and loadings biplot, loaded "
           "from the same measurements database "
           "(spacr.qt.screens.image_umap)",
    # The segmentation workbench's four. Make Masks is the screen a user is
    # already on when they want any of them -- segment, look, correct, train,
    # segment again is one loop -- so each is a button on its masthead that
    # opens the module's own screen as a page beside the editor.
    "train_cellpose": "a button on the Make Masks masthead that opens the "
                      "Cellpose Workbench as a page, on its Train tab with "
                      "its own path untouched (spacr.qt.screens.make_masks)",
    "model_compare": "a button on the Make Masks masthead that opens the A/B "
                     "harness as a page, pointed at the open folder; Model "
                     "Zoo's compare hand-off opens the same one "
                     "(spacr.qt.screens.make_masks)",
    "model_zoo": "a button on the Make Masks masthead that opens the browser "
                 "as a page, benching models on the open folder's fields "
                 "(spacr.qt.screens.make_masks)",
    "curate": "a button on the Make Masks masthead that opens the brush and "
              "the track surgery as a page, handed the field on screen -- and "
              "the fold is where Curate got the Save-mask it never had "
              "(spacr.qt.screens.make_masks)",
    "volcano_explorer": "\u201cPublication figure\u2026\u201d on the "
                        "Regression volcano's own right-click menu, and a "
                        "button on that masthead; both open the explorer "
                        "seeded with the frame on screen "
                        "(spacr.qt.screens.regression)",
    "hit_list": "a tab on the Regression results and a button on that "
                "masthead; the ranked, annotated, filterable hits are read "
                "off the run already on screen "
                "(spacr.qt.screens.regression)",
    "methods_export": "a button on the Regression masthead that opens the "
                      "methods-and-results draft for the run on screen, with "
                      "every number traced back to the digest "
                      "(spacr.qt.screens.regression)",
    "napari_bridge": "a button on the Make Masks masthead that hands the "
                     "field on screen to napari and reads the corrected "
                     "labels back (spacr.qt.screens.make_masks)",
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


def live_app_keys():
    """Every app key, including screens that register themselves on import.

    :data:`APP_KEYS` above is the table inside ``app.py``, snapshotted when
    this module is imported. Since the registration seam landed, a screen may
    own its row instead — ``register_app`` at import time — and those rows
    only exist once ``spacr.qt.screens`` has been imported, which importing
    ``spacr.qt.app`` does not do. A question about whether a key names a
    *real* app therefore has to ask the live registry; asked of the snapshot,
    every seam-registered app looks like a ghost.
    """
    import spacr.qt.screens                # noqa: F401 - the import registers
    from spacr.qt.app import APPS as LIVE
    return {row[0] for row in LIVE}


def test_the_gui_only_list_holds_no_apps_that_no_longer_exist():
    """A stale entry hides a genuinely unknown module behind a helpful lie.

    A folded module is exempt and named in :data:`FOLDED`: its sentence is
    the only thing `spacr-run <key>` can say, and it is more useful after
    the fold than before -- "run it in spacr-qt" is a worse answer than
    "it is a button on the Annotate masthead".
    """
    ghosts = sorted(set(cli.INTERACTIVE_ONLY) - live_app_keys() - set(FOLDED))
    assert not ghosts, (
        f"cli.INTERACTIVE_ONLY names apps that are not in spacr.qt.app.APPS "
        f"and are not recorded as folded: {ghosts}")


def test_every_cli_module_is_an_app_or_a_declared_headless_only_one():
    """The other direction: a module nobody can reach from the GUI."""
    orphans = sorted(set(cli.MODULES) - set(APP_KEYS) - set(HEADLESS_ONLY)
                       - set(FOLDED))
    assert not orphans, (
        f"spacr.cli.MODULES has modules with no APPS entry and no reason "
        f"recorded in this test's HEADLESS_ONLY or FOLDED: {orphans}")
    # Aliases count as existing: a retired key that was pointed at its
    # successor still resolves from the command line, and recording why it
    # has no button is exactly what this table is for.
    stale = sorted(set(HEADLESS_ONLY) - set(cli.MODULES) - set(cli.ALIASES))
    assert not stale, f"HEADLESS_ONLY names modules that no longer exist: {stale}"
    for key in HEADLESS_ONLY:
        assert cli.resolve_module(key) is not None, (
            f"`spacr-run {key}` does not resolve, so HEADLESS_ONLY's claim "
            f"that it runs headless is not true")


def test_every_folded_module_really_is_folded_and_really_is_reachable():
    """FOLDED is an exemption, so it has to be paid for in both directions.

    A key here must be genuinely OUT of the registry -- otherwise the
    exemption is dead text hiding nothing -- and genuinely IN some host's
    fold table, which is what makes it something a user can still press.
    A key that is in neither is the failure this whole exercise exists to
    avoid: a module folded out of the GUI and into nothing.
    """
    import importlib

    import spacr.qt.screens                            # noqa: F401
    from spacr.qt.screens import make_masks
    from spacr.qt.widgets.fold_strip import FOLD_HOST_MODULES

    live = live_app_keys()
    # EVERY host, DERIVED rather than listed. A key folded onto a host
    # missing from this union reads as "folded into nothing", which is the
    # failure below -- and it would be reported against the module rather
    # than against the list that had fallen behind.
    #
    # This WAS a hand-written tuple of seven screens, and its own comment
    # said it should be every host. It was not: Align & Stitch grew a fold
    # and the tuple did not follow, so a correctly wired module failed here
    # for a reason that had nothing to do with it. `FOLD_HOST_MODULES` is
    # the same list the application itself reads, so the two cannot now
    # disagree.
    hosted = set()
    for module_name in FOLD_HOST_MODULES:
        try:
            host = importlib.import_module(module_name)
        except Exception:                              # noqa: BLE001
            continue
        hosted |= set(getattr(host, "FOLDED_APPS", ()))
    hosted |= set(make_masks.FOLD_ORDER)

    still_registered = sorted(set(FOLDED) & live)
    assert not still_registered, (
        f"FOLDED names apps that still have a registry row: "
        f"{still_registered}")
    unreachable = sorted(set(FOLDED) - hosted)
    assert not unreachable, (
        f"FOLDED names modules no host screen offers, so nothing in the "
        f"GUI can open them: {unreachable}")
    assert all(reason.strip() for reason in FOLDED.values())


def test_a_folded_module_still_answers_everywhere_it_used_to():
    """The row went; the module did not.

    Each folded key is checked against the four tables a run actually
    reads -- the CLI module list or the GUI-only sentence, the pre-flight
    entry, the pipeline entry point and the settings defaults -- because
    "it is a button now" is only true if pressing it, or naming it on the
    command line, still starts the same work.
    """
    from spacr.qt.bridge import resolve_pipeline_entry
    from spacr.qt.screens.settings_model import resolve_default_settings

    for key in FOLDED:
        headless = key in cli.MODULES
        assert headless or key in cli.INTERACTIVE_ONLY, (
            f"{key} can neither run headless nor say why not")
        if not headless:
            continue
        assert key in APP_FUNCTIONS, f"{key} lost its pre-flight entry"
        assert resolve_pipeline_entry(key) is not None, (
            f"{key} has no entry point, so its Run button is dead")
        defaults = resolve_default_settings(key)
        assert len(defaults) > 1, (
            f"{key} resolves to an empty settings form: {defaults}")


def test_every_validate_entry_is_an_app_or_a_headless_only_module():
    orphans = sorted(set(APP_FUNCTIONS) - set(APP_KEYS) - set(HEADLESS_ONLY)
                       - set(FOLDED))
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
    """--describe has to answer even against a half-installed environment.

    The two halves are deliberately different, and this test used to assert
    only the first and never call describe at all. ``module_defaults`` is the
    **run** path: it once returned ``{}`` here, which made every module with
    its own helper -- convert, illumination, foreign, external_masks,
    barcode_qc, anndata_export, every plugin app -- run on a settings dict
    with no defaults in it, so ``--set z_handling=max`` came back as "names a
    setting that does not exist for module 'convert'". It now raises and names
    the dependency. ``--describe`` keeps its own guard around that call, so it
    still prints the contract it can work out without the helper.
    """
    broken = cli.Module(key="_broken", summary="", entry="spacr.foreign:import_project",
                        defaults=None, validate_key="",
                        defaults_entry="spacr_no_such_module:default_settings")

    # The run path fails loudly, pointing at the dependency and not at the
    # user's command line.
    with pytest.raises(cli.SettingsError) as excinfo:
        cli.module_defaults(broken)
    message = str(excinfo.value)
    assert "_broken" in message and "spacr_no_such_module" in message
    assert "will not import" in message

    # The describe path still answers, and answers usefully.
    described = cli.render_module_description(broken)
    assert broken.defaults_label == "spacr_no_such_module.default_settings()"
    assert "spacr_no_such_module.default_settings()" in described
    assert "spacr.foreign.import_project(settings)" in described
    # ...without inventing a settings count it could not compute.
    assert "keys, all optional" not in described


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
