"""Pre-flight rules that only fire on the settings nobody usually sends.

:mod:`spacr.validate` reads a settings dict and the folder it points at and
says what would go wrong, without starting the run. The rules exercised here
are the ones a healthy settings dict never reaches: the per-app blocks for
Foreign Import, External Masks, Explain CV, Investigate Hit and Replication,
the numeric-range guards, and the paths taken when the input on disk is
malformed rather than absent.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from spacr import validate as V


def _plate(tmp_path, name="plate1", *, merged=0, stacks=0, planes=4, raw=()):
    """A plate folder holding the arrays and raw files asked for."""
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    if merged:
        merged_dir = root / "merged"
        merged_dir.mkdir(exist_ok=True)
        for i in range(merged):
            np.save(merged_dir / f"field_{i}.npy",
                    np.zeros((8, 8, planes), dtype=np.uint16))
    if stacks:
        stack_dir = root / "stack"
        stack_dir.mkdir(exist_ok=True)
        for i in range(stacks):
            np.save(stack_dir / f"field_{i}.npy",
                    np.zeros((8, 8, planes), dtype=np.uint16))
    for filename in raw:
        (root / filename).write_bytes(b"")
    return str(root)


def _keys(problems):
    return {p.setting for p in problems}


def _messages(problems):
    return "\n".join(p.message for p in problems)


# ---------------------------------------------------------------------------
# app-key normalisation and the unknown-app warning
# ---------------------------------------------------------------------------

def test_a_non_string_app_key_is_no_app_at_all():
    """A key that is not text normalises to the empty app, not to a crash."""
    assert V._normalize_app(None) == ""
    assert V._normalize_app(17) == ""
    assert V._normalize_app("  Cellpose_All ") == "cellpose_masks"


def test_settings_that_are_not_a_dict_are_reported_not_raised():
    """The one problem returned names the type that was passed instead."""
    problems = V.validate_settings(["src"], "mask")
    assert len(problems) == 1 and problems[0].is_error
    assert "got list" in problems[0].message
    assert V.describe_plan(["src"], "mask").startswith("Plan unavailable")


# ---------------------------------------------------------------------------
# reading what is on disk
# ---------------------------------------------------------------------------

def test_a_zero_dimensional_array_is_skipped_when_counting_planes(tmp_path):
    """A shapeless ``.npy`` is passed over for the next one, not believed.

    ``shape[-1]`` on a 0-d array raises, so the scan has to move on; the
    plane count then comes from a real field.
    """
    merged = tmp_path / "plate" / "merged"
    merged.mkdir(parents=True)
    np.save(merged / "aaa_scalar.npy", np.array(3, dtype=np.uint16))
    np.save(merged / "bbb_field.npy", np.zeros((4, 4, 5), dtype=np.uint16))

    planes, example, count = V._peek_planes(str(merged))
    assert planes == 5
    assert example == "bbb_field.npy"
    assert count == 2


def test_an_uncompilable_custom_regex_does_not_stop_the_scan(tmp_path):
    """A broken ``custom_regex`` is skipped; the built-in patterns still run."""
    src = _plate(tmp_path, raw=("plate1_A01_T0001F001L01A01Z01C01.tif",
                                "plate1_A01_T0001F001L01A01Z01C02.tif"))
    inv = V._Inventory()
    V._scan_raw_images(src, {"custom_regex": "(?P<unclosed"}, inv)
    assert inv.raw_files == 2
    assert inv.raw_channels == 2
    assert inv.regex_used and inv.regex_used != "custom_regex"


def test_a_pattern_without_a_channel_group_scores_no_hits(tmp_path):
    """A regex that matches but names no ``chanID`` contributes nothing.

    Channel identity is the whole point of the scan, so a match that cannot
    supply one must not be counted as evidence for a channel count.
    """
    src = _plate(tmp_path, raw=("img_one.tif", "img_two.tif"))
    inv = V._Inventory()
    V._scan_raw_images(src, {"custom_regex": r"(?P<chanID>x)?img_.*"}, inv)
    assert inv.raw_files == 2
    assert inv.raw_channels is None


def test_a_src_that_looks_like_a_list_but_is_not_one_stays_one_path():
    """``[not, python]`` is treated as the literal path it is."""
    assert V._src_values({"src": "[a, b]"}) == ["[a, b]"]
    assert V._src_values({"src": "['/one', '/two']"}) == ["/one", "/two"]


def test_external_mask_groups_are_reduced_to_the_folders_they_live_in(tmp_path):
    """Dropped files collapse to their folder; a named root wins over paths."""
    one = tmp_path / "images"
    one.mkdir()
    dropped = one / "a.tif"
    dropped.write_bytes(b"")
    values = V._src_values(
        {"inputs": [
            {"root": str(tmp_path / "explicit")},
            {"paths": [str(dropped), str(one)]},
            str(dropped),
            {"paths": []},
        ]},
        "external_masks")
    assert values == [str(tmp_path / "explicit"), str(one)]


def test_a_database_file_as_src_is_not_reported_as_the_wrong_kind_of_thing(tmp_path):
    """Pointing an analysis app straight at ``measurements.db`` is allowed."""
    db = tmp_path / "measurements.db"
    db.write_bytes(b"SQLite format 3\x00")
    problems = V._check_src({"src": str(db)}, "umap",
                            [V._inventory(str(db), {}, "umap")])
    assert not [p for p in problems if "not a folder" in p.message]


def test_a_src_list_entry_that_is_not_a_path_string_is_named(tmp_path):
    """Each non-string entry is reported with the type that was found."""
    problems = V._check_src({"src": [7, None]}, "mask",
                            [V._inventory(7, {}, "mask")])
    messages = _messages(problems)
    assert "is a int, not a path string" in messages
    assert "is a NoneType, not a path string" in messages


# ---------------------------------------------------------------------------
# channel indices
# ---------------------------------------------------------------------------

def test_a_boolean_channel_index_is_not_read_as_zero_or_one():
    """``True`` is not an index; ``_as_int`` refuses it rather than saying 1."""
    assert V._as_int(True) is None
    assert V._as_int(False) is None
    assert V._as_int(3) == 3
    assert V._as_int(4.0) == 4
    assert V._as_int(4.5) is None


def test_a_channel_past_the_stored_planes_is_reported_when_no_raw_count_exists(tmp_path):
    """With only merged arrays to go on, the plane count is the ceiling."""
    src = _plate(tmp_path, merged=1, planes=3)
    settings = {"src": src, "cell_channel": 7}
    problems = V.validate_settings(settings, "mask")
    channel = [p for p in problems if p.setting == "cell_channel"]
    assert channel and channel[0].is_error
    assert "past the end of the stored arrays" in channel[0].message


def test_a_negative_mask_dim_is_an_error_and_a_vague_one_is_skipped(tmp_path):
    """Mask planes are zero-based positions; ``None`` skips the object."""
    src = _plate(tmp_path, merged=1, planes=6)
    problems = V.validate_settings(
        {"src": src, "cell_mask_dim": -1, "nucleus_mask_dim": "later",
         "pathogen_mask_dim": None},
        "measure")
    negative = [p for p in problems if p.setting == "cell_mask_dim"]
    assert negative and negative[0].is_error and "is negative" in negative[0].message
    assert not [p for p in problems if p.setting == "pathogen_mask_dim"]


# ---------------------------------------------------------------------------
# type and numeric sanity
# ---------------------------------------------------------------------------

def test_a_tuple_is_accepted_wherever_a_list_is_declared():
    """``(0, 1)`` for a list-typed setting is a spelling, not a mistake."""
    problems = V._check_types({"channels": (0, 1, 2)}, "measure")
    assert not [p for p in problems if p.setting == "channels"]


def test_a_settings_key_that_is_not_a_string_is_skipped_by_the_range_checks():
    """A non-text key cannot be matched by name, so it is passed over."""
    assert V._check_numeric_sanity({7: 100.0}) == []


@pytest.mark.parametrize("key,value,severity", [
    ("flow_threshold", 9.0, V.WARNING),
    ("val_split", 4.0, V.ERROR),
    ("dropout_rate", -0.5, V.ERROR),
    ("learning_rate", 0.0, V.ERROR),
])
def test_a_number_outside_its_useful_range_is_reported(key, value, severity):
    """Each guarded numeric key states its own range in the message."""
    problems = V._check_numeric_sanity({key: value})
    matching = [p for p in problems if p.setting == key]
    assert matching, f"{key}={value} produced no problem"
    assert matching[0].severity == severity


def test_a_percentile_outside_zero_to_one_hundred_is_an_error():
    """Percentile lists are percentages, so 0-100 is the whole range."""
    problems = V._check_numeric_list("normalize", [5, 140])
    assert problems and problems[0].is_error
    assert "outside 0-100" in problems[0].message


def test_a_percentile_list_with_an_unreadable_entry_is_left_alone():
    """One unparseable entry means the list is not a percentile pair to judge."""
    assert V._check_numeric_list("normalize", ["low", 99]) == []
    assert V._check_numeric_list("normalize", []) == []


def test_a_png_size_of_zero_pixels_is_an_error():
    """Crops are sized in pixels, so a non-positive edge cannot be honoured."""
    problems = V._check_numeric_list("png_size", [[224, 0]])
    assert problems and problems[0].is_error
    assert "non-positive size" in problems[0].message


# ---------------------------------------------------------------------------
# per-app required inputs
# ---------------------------------------------------------------------------

def test_a_foreign_import_names_every_input_it_was_not_given(tmp_path):
    """Missing, blank and non-existent inputs each get their own problem."""
    problems = V._check_required_paths(
        {"images": "", "masks": [], "measurements": str(tmp_path / "nope.csv")},
        "foreign")
    assert _keys(problems) >= {"images", "masks", "measurements", "column_map"}
    text = _messages(problems)
    assert "images is not set" in text
    assert "masks is not set" in text
    assert "points at a path that does not exist" in text
    assert "no reviewed column_map" in text


def test_a_foreign_import_previewing_its_columns_is_not_nagged_about_them(tmp_path):
    """``preview_only`` is how the column map gets reviewed in the first place."""
    real = tmp_path / "images"
    real.mkdir()
    problems = V._check_required_paths(
        {"images": str(real), "masks": str(real), "measurements": str(real),
         "preview_only": True},
        "foreign")
    assert problems == []


def test_external_masks_with_nothing_dropped_says_so_once():
    """No inputs at all is one problem, not one per missing role."""
    problems = V._check_required_paths({"inputs": []}, "external_masks")
    assert len(problems) == 1
    assert "No external images or masks" in problems[0].message


def test_external_masks_names_the_role_and_the_object_type_that_are_missing():
    """Images, masks and each mask's object type are three separate rules."""
    problems = V._check_required_paths(
        {"inputs": [{"role": "mask", "object_type": None},
                    {"role": "mask", "object_type": "cell"}]},
        "external_masks")
    text = _messages(problems)
    assert "No input group is assigned as images" in text
    assert "1 mask group(s) have no object type" in text
    assert "No input group is assigned as masks" not in text


def test_a_unet_organelle_without_a_model_file_is_an_error(tmp_path):
    """``organelle_method='unet'`` needs a serialised net that exists."""
    role = V.SEGMENTED_ROLES[3]
    missing = str(tmp_path / "absent.pth")
    problems = V._check_required_paths(
        {f"{role}_method": "unet", f"{role}_unet_model_path": missing}, "mask")
    assert problems and problems[0].is_error
    assert "not found" in problems[0].message

    blank = V._check_required_paths(
        {f"{role}_method": "unet", f"{role}_unet_model_path": "  "}, "mask")
    assert blank and "but no" in blank[0].message


def test_explain_cv_refuses_an_unknown_surrogate_family_and_split(tmp_path):
    """The surrogate vocabulary is closed, and both inputs must exist."""
    problems = V._check_app_specific(
        {"db_path": "", "predictions_file": str(tmp_path / "gone.csv"),
         "surrogate_model": "MagicForest", "surrogate_split_by": "cell"},
        "explain_cv")
    text = _messages(problems)
    assert "no measurements database is selected" in text
    assert "prediction CSV does not exist" in text
    assert "unsupported surrogate family 'magicforest'" in text
    assert "unsupported split unit 'cell'" in text


def test_investigate_hit_requires_the_exact_files_it_was_handed(tmp_path):
    """The investigation never infers newest files, so each one is checked."""
    problems = V._check_app_specific(
        {"db_path": "", "predictions_file": "", "guide_fractions_file": "",
         "results_folder": "", "target_gene": "", "target_guides": [],
         "hit_direction": "sideways"},
        "investigate_hit")
    assert _keys(problems) == {"db_path", "predictions_file",
                               "guide_fractions_file", "results_folder",
                               "target_guides", "hit_direction"}
    assert all(p.is_error for p in problems)


def test_a_replication_run_is_held_to_the_doubling_ladder():
    """Buckets follow endodyogeny, so the maximum has to be a power of two."""
    problems = V._check_app_specific(
        {"max_parasites_per_vacuole": 6, "vacuole_link_factor": 0,
         "vacuole_link_distance": -3, "non_power_of_two_warn": 1.5,
         "min_parasite_area": 90, "max_parasite_area": 10},
        "replication")
    assert _keys(problems) == {"max_parasites_per_vacuole",
                               "vacuole_link_factor", "vacuole_link_distance",
                               "non_power_of_two_warn", "max_parasite_area"}
    assert "not a positive power of two" in _messages(problems)


def test_a_replication_run_on_the_ladder_is_left_alone():
    """Eight parasites, a positive linking distance and a real fraction pass."""
    assert V._check_app_specific(
        {"max_parasites_per_vacuole": 8, "vacuole_link_factor": 1.5,
         "vacuole_link_distance": 12, "non_power_of_two_warn": 0.2,
         "min_parasite_area": 10, "max_parasite_area": 900},
        "replication") == []


def test_a_pre_sam_pathogen_model_is_called_out_as_ignored():
    """Cellpose 4 ships only ``cpsam``; the old checkpoints were removed."""
    problems = V._check_app_specific(
        {"pathogen_channel": 2, "pathogen_model": "toxo_pv_lumen"}, "mask")
    matching = [p for p in problems if p.setting == "pathogen_model"]
    assert matching and matching[0].severity == V.WARNING
    assert "cpsam" in matching[0].message


def test_more_dilation_ratios_than_crop_modes_are_reported_as_ignored():
    """Extra entries are silently dropped by the pipeline, so say so first."""
    problems = V._check_app_specific(
        {"crop_mode": ["cell"], "dialate_png_ratios": [0.2, 0.3, 0.4],
         "dialate_pngs": True, "cell_mask_dim": 4, "save_png": True},
        "measure")
    matching = [p for p in problems if p.setting == "dialate_png_ratios"]
    assert matching and matching[0].severity == V.WARNING
    assert "the extras are ignored" in matching[0].message


# ---------------------------------------------------------------------------
# plugin-supplied apps and validators
# ---------------------------------------------------------------------------

def test_a_plugin_registry_that_cannot_be_read_still_validates_the_built_ins(
        monkeypatch):
    """A broken plugin lookup must not silence the generic checks."""
    import spacr.plugins as P

    def boom(_key):
        raise RuntimeError("plugin metadata is corrupt")

    monkeypatch.setattr(P, "get_app", boom)
    problems = V.validate_settings({"src": ""}, "not_an_app")
    text = _messages(problems)
    assert "unknown app 'not_an_app'" in text
    assert "Plugin validation for 'not_an_app' failed" in text


def _plugin_app(**overrides):
    from spacr.plugins import AppContribution

    fields = dict(key="probe_app", name="Probe", description="a probe",
                  entrypoint="spacr.validate:validate_settings",
                  defaults="spacr.settings:get_measure_crop_settings")
    fields.update(overrides)
    return AppContribution(**fields)


def test_a_plugin_validator_may_return_problems_or_the_mappings_for_them(
        monkeypatch):
    """Both shapes are accepted, and a bare Problem counts as one result."""
    import spacr.plugins as P

    def validator(settings):
        return [V.Problem(V.WARNING, "probe", "a mapping follows", "none"),
                {"severity": V.ERROR, "setting": "probe",
                 "message": "from a mapping", "fix": "none"}]

    monkeypatch.setattr(P, "get_app",
                        lambda key: _plugin_app(validator="probe:validator")
                        if key == "probe_app" else None)
    monkeypatch.setattr(P, "load_object", lambda ref: validator)

    problems = V.validate_settings({"src": ""}, "probe_app")
    text = _messages(problems)
    assert "a mapping follows" in text
    assert "from a mapping" in text
    assert "Plugin validation" not in text


def test_a_plugin_validator_returning_nothing_adds_nothing(monkeypatch):
    """``None`` is the plugin saying it found no problem."""
    import spacr.plugins as P

    monkeypatch.setattr(P, "get_app",
                        lambda key: _plugin_app(validator="probe:validator")
                        if key == "probe_app" else None)
    monkeypatch.setattr(P, "load_object", lambda ref: (lambda settings: None))
    assert not [p for p in V.validate_settings({"src": ""}, "probe_app")
                if "Plugin validation" in p.message]


def test_a_single_problem_from_a_plugin_validator_is_not_iterated_apart(
        monkeypatch):
    """One Problem is one result, not four fields to walk over."""
    import spacr.plugins as P

    monkeypatch.setattr(P, "get_app",
                        lambda key: _plugin_app(validator="probe:validator")
                        if key == "probe_app" else None)
    monkeypatch.setattr(
        P, "load_object",
        lambda ref: (lambda settings: V.Problem(V.ERROR, "probe",
                                                "single problem", "fix it")))
    text = _messages(V.validate_settings({"src": ""}, "probe_app"))
    assert "single problem" in text
    assert "Plugin validation" not in text


@pytest.mark.parametrize("result,expected", [
    ("not a validator at all", "is not callable"),
    (None, "must be Problem objects or mappings"),
])
def test_a_plugin_validator_that_misbehaves_is_reported_as_a_plugin_failure(
        monkeypatch, result, expected):
    """A bad plugin stops the run and says which plugin, not what crashed."""
    import spacr.plugins as P

    monkeypatch.setattr(P, "get_app",
                        lambda key: _plugin_app(validator="probe:validator")
                        if key == "probe_app" else None)
    if result is None:
        monkeypatch.setattr(P, "load_object",
                            lambda ref: (lambda settings: ["a bare string"]))
    else:
        monkeypatch.setattr(P, "load_object", lambda ref: result)

    problems = V.validate_settings({"src": ""}, "probe_app")
    failures = [p for p in problems if "Plugin validation" in p.message]
    assert failures and failures[0].is_error
    assert expected in failures[0].message


# ---------------------------------------------------------------------------
# apps whose source setting is not called src
# ---------------------------------------------------------------------------

def test_a_foreign_import_is_told_about_images_not_about_src():
    """``foreign`` reads ``images``, so that is the key the fix names."""
    problems = V._check_src({"images": ""}, "foreign", [])
    assert len(problems) == 1
    assert problems[0].setting == "images"
    assert problems[0].fix == "Set images to the folder holding their images."


def test_external_masks_with_images_but_no_masks_says_which_role_is_missing():
    """Only the role actually absent is reported."""
    problems = V._check_required_paths(
        {"inputs": [{"role": "image"}]}, "external_masks")
    text = _messages(problems)
    assert "No input group is assigned as masks" in text
    assert "No input group is assigned as images" not in text


# ---------------------------------------------------------------------------
# the known-key universe
# ---------------------------------------------------------------------------

def test_a_settings_helper_that_refuses_both_call_shapes_is_skipped(monkeypatch):
    """A helper that raises with and without an argument contributes no keys.

    The sweep calls every ``get_*``/``set_*`` helper to learn the key
    universe. One that cannot be called at all must be stepped over, or a
    single broken helper would take typo detection down with it.
    """
    from spacr import settings as S

    def get_helper_that_cannot_be_called(*args, **kwargs):
        raise RuntimeError("this helper needs a live database")

    monkeypatch.setattr(S, "get_helper_that_cannot_be_called",
                        get_helper_that_cannot_be_called, raising=False)
    monkeypatch.setattr(V, "_KNOWN_KEYS_CACHE", None)

    keys = V._known_setting_keys()
    assert "src" in keys
    assert isinstance(keys, frozenset) and len(keys) > 50


# ---------------------------------------------------------------------------
# plugin apps joining the alias table at import
# ---------------------------------------------------------------------------

def _reimport_validate(monkeypatch, plugin_apps):
    """Execute ``spacr/validate.py`` again with a chosen plugin registry.

    Reloading the shipped module would hand later tests a second
    :class:`Problem` class, so the file is executed under a private name
    inside the same package instead. Relative imports resolve because the
    parent package is still ``spacr``.
    """
    import importlib.util
    import sys

    import spacr.plugins as P

    monkeypatch.setattr(P, "plugin_apps", plugin_apps)
    name = "spacr._validate_under_test"
    spec = importlib.util.spec_from_file_location(name, V.__file__)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_a_plugin_app_joins_the_function_table_and_the_alias_table(monkeypatch):
    """Its key and every alias resolve the way a built-in app's do."""
    contribution = _plugin_app(key="probe_app", aliases=("Probe-App", "probe"))
    module = _reimport_validate(monkeypatch, lambda: (contribution,))

    assert module.APP_FUNCTIONS["probe_app"] == "spacr.validate.validate_settings"
    assert module.APP_ALIASES["probe_app"] == "probe_app"
    assert module.APP_ALIASES["probe"] == "probe_app"
    assert module._normalize_app("  PROBE ") == "probe_app"


@pytest.mark.xfail(strict=True, reason=(
    "validate.py:167 registers a plugin alias with '-' folded to '_', but "
    "_normalize_app (line 239) only strips and lowercases, so the hyphenated "
    "spelling the plugin declared never resolves"))
def test_a_hyphenated_plugin_alias_resolves_by_the_spelling_it_declared(
        monkeypatch):
    """An alias is reachable by the text the plugin wrote down.

    Registration folds ``-`` to ``_`` so both spellings share one entry; the
    lookup has to fold the caller's key the same way or the fold only ever
    hides the declared spelling.
    """
    contribution = _plugin_app(key="probe_app", aliases=("probe-app",))
    module = _reimport_validate(monkeypatch, lambda: (contribution,))
    assert module._normalize_app("probe-app") == "probe_app"


def test_plugin_discovery_that_raises_leaves_the_built_in_table_intact(
        monkeypatch):
    """Built-in validation stays useful when third-party metadata cannot load."""
    def boom():
        raise RuntimeError("entry points are unreadable")

    module = _reimport_validate(monkeypatch, boom)
    assert "mask" in module.APP_FUNCTIONS
    assert module.APP_FUNCTIONS.keys() == V.APP_FUNCTIONS.keys()


# ---------------------------------------------------------------------------
# the plan card
# ---------------------------------------------------------------------------

def test_a_plan_for_a_mask_run_names_the_model_and_the_channel_stacks(tmp_path):
    """Channel stacks with no merged arrays are still counted as input."""
    src = _plate(tmp_path, stacks=2, planes=3)
    plan = V.describe_plan(
        {"src": src, "pathogen_channel": 1, "pathogen_model": "cpsam",
         "n_jobs": 4},
        "mask")
    assert "2 channel stacks (.npy)" in plan
    assert "model cpsam" in plan
    assert "~2 field stacks" in plan
    assert "n_jobs=4" in plan


def test_a_plan_counts_raw_files_no_regex_could_read(tmp_path):
    """Files the metadata patterns miss are reported as files, not fields."""
    src = _plate(tmp_path, raw=("unreadable_one.tif", "unreadable_two.tif"))
    plan = V.describe_plan({"src": src, "custom_regex": "will_not_match"},
                           "mask")
    assert "2 raw image files" in plan
    assert "~2 image files" in plan


def test_a_measure_plan_names_the_cytoplasm_and_the_crop_settings(tmp_path):
    """Cytoplasm is derived rather than segmented, and says so."""
    src = _plate(tmp_path, merged=1, planes=5)
    plan = V.describe_plan(
        {"src": src, "cell_mask_dim": 4, "cytoplasm": True, "save_png": True,
         "crop_mode": ["cell"], "png_size": [224, 224], "channels": None},
        "measure")
    assert "cytoplasm: derived from the cell mask" in plan
    assert "cell PNGs at [224, 224]" in plan
    assert "measures channels" in plan and "not set" in plan


def test_an_analysis_plan_says_which_database_it_would_read(tmp_path):
    """A database app reads ``measurements/measurements.db`` under ``src``."""
    src = _plate(tmp_path)
    plan = V.describe_plan({"src": src}, "umap")
    assert "would read" in plan
    assert os.path.join("measurements", "measurements.db") in plan


def test_a_report_over_several_sources_counts_the_ones_it_did_not_name():
    """The header names the first source and how many more there were."""
    text = V.format_report([], {"src": ["/one", "/two", "/three"]}, "mask")
    assert "/one (+2 more)" in text


def test_a_value_is_shown_as_written_or_as_not_set():
    """``_fmt`` prints lists in brackets and ``None`` as words."""
    assert V._fmt(None) == "not set"
    assert V._fmt([0, 1]) == "[0, 1]"
    assert V._fmt((2, 3)) == "[2, 3]"
    assert V._fmt(0.4) == "0.4"


# ---------------------------------------------------------------------------
# the resource card
# ---------------------------------------------------------------------------

def test_a_source_that_is_not_on_disk_leaves_the_intensity_scan_out(tmp_path):
    """With no merged or stack folder there is nothing to scan, and no note."""
    card = V.describe_resources({"src": str(tmp_path / "absent")}, "measure")
    assert "INTENSITY PREFLIGHT" not in card


def test_an_intensity_scan_that_fails_is_reported_not_swallowed(
        tmp_path, monkeypatch):
    """A scan that cannot run is named, because the run's scaling depends on it."""
    from spacr import intensity_rescale as IR

    def boom(directory, names, settings):
        raise RuntimeError("the plate could not be read")

    monkeypatch.setattr(IR, "build_plate_plan", boom)
    src = _plate(tmp_path, merged=2, planes=4)
    card = V.describe_resources({"src": src}, "measure")
    assert "INTENSITY PREFLIGHT: the plate-wide scan could not be completed" in card
    assert "the plate could not be read" in card


def test_a_preflight_prints_the_report_the_plan_and_the_resources(tmp_path):
    """All three cards reach the printer, and the problems come back."""
    src = _plate(tmp_path, merged=1, planes=4)
    lines = []
    problems = V.run_preflight({"src": src, "cell_mask_dim": 99}, "measure",
                               printer=lines.append, trailer="the trailer")
    text = "\n".join(lines)
    assert "spaCR pre-flight check" in text
    assert "Plan — what this run would do" in text
    assert "the trailer" == lines[-1]
    assert any(p.setting == "cell_mask_dim" for p in problems)


def test_a_resource_card_that_raises_does_not_deny_the_rest_of_the_preflight(
        monkeypatch, tmp_path):
    """The projection is best-effort; the report and plan are not."""
    monkeypatch.setattr(
        V, "describe_resources",
        lambda settings, app_key="": (_ for _ in ()).throw(
            RuntimeError("the disk went away")))
    lines = []
    V.run_preflight({"src": str(tmp_path)}, "mask", printer=lines.append,
                    trailer="")
    text = "\n".join(lines)
    assert "Resources — could not be projected: the disk went away" in text
    assert "Plan — what this run would do" in text
