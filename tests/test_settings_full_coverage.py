"""Every remaining branch of spacr.settings, pinned by behaviour.

F16. settings.py is where an omission is invisible: a key that is never
defaulted does not raise, it just makes the module downstream of it do
nothing. `summarize_organelles_by` was missing from the measure defaults and
the result was a measure run that wrote no organelle table at all, with no
error anywhere. So these tests assert what each default *means* — the value,
the type it will survive `check_settings` as, and the behaviour it selects —
rather than that a function returned a dict.
"""
from __future__ import annotations

import inspect
import os
import sys
import types

import pytest

import spacr.settings as S


# ---------------------------------------------------------------------------
# bundled_barcode_path — the shipped barcode references
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["column", "grna", "row"])
def test_bundled_barcode_path_points_at_a_real_reference_csv(kind):
    path = S.bundled_barcode_path(kind)
    assert os.path.isabs(path)
    assert os.path.isfile(path), f"the shipped {kind} barcode CSV is missing"
    import pandas as pd
    df = pd.read_csv(path)
    assert {"sequence", "name"}.issubset(df.columns)


@pytest.mark.parametrize("kind", ["column", "row", pytest.param(
    "grna",
    marks=pytest.mark.xfail(
        strict=True,
        reason=(
            "The shipped barcodes_grna.csv contains three protospacers that "
            "each appear under two or three different gene names "
            "(GCCGGCGATAGAGCCCCGCCC -> TGGT1_241310_2 / TGGT1_411210_2 / "
            "TGGT1_411710_2; GCGATAGAGCCCCGCCCTGG -> TGGT1_411210_3 / "
            "TGGT1_411710_3; GTCGCTAGGACATCCTCCAAG -> TGGT1_241310_10 / "
            "TGGT1_411210_10 / TGGT1_411710_10). "
            "map_sequences_to_names refuses a reference with duplicate "
            "sequences -- correctly, since one read would otherwise be "
            "assigned to whichever name the dict happened to keep -- so "
            "generate_barecode_mapping raises ValueError on its first chunk "
            "when run with the SHIPPED defaults and nothing else changed. "
            "Fixing it means deciding which gene a shared protospacer counts "
            "for, which is a screen-design call, not a code change, and the "
            "data file is outside this task's edit scope."))),
])
def test_the_shipped_barcode_references_load_through_the_pipeline(kind):
    """The shipped default must be loadable by the code that loads it."""
    from spacr.sequencing import map_sequences_to_names
    path = S.bundled_barcode_path(kind)
    assert map_sequences_to_names(path, ["ACGTACGT"], rc=False) == [
        __import__("pandas").NA]


def test_bundled_barcode_path_is_case_insensitive():
    assert S.bundled_barcode_path("COLUMN") == S.bundled_barcode_path("column")


def test_bundled_barcode_path_names_the_valid_kinds(capsys):
    with pytest.raises(ValueError) as exc:
        S.bundled_barcode_path("plate")
    msg = str(exc.value)
    assert "Unknown barcode reference 'plate'" in msg
    # The message lists what the caller could have said instead.
    for kind in ("column", "grna", "row"):
        assert kind in msg


def test_the_three_bundled_references_are_three_different_files():
    paths = {S.bundled_barcode_path(k) for k in ("column", "grna", "row")}
    assert len(paths) == 3


# ---------------------------------------------------------------------------
# _default_worker_count — the default that went to zero on CI
# ---------------------------------------------------------------------------

def test_default_worker_count_reserves_cores(monkeypatch):
    monkeypatch.setattr(S.os, "cpu_count", lambda: 16)
    assert S._default_worker_count(reserve=4) == 12
    assert S._default_worker_count() == 16


@pytest.mark.parametrize("cores", [None, 1, 2, 4])
def test_default_worker_count_never_returns_zero(monkeypatch, cores):
    """The shipped mask default went to 0 on a 4-core runner and preflight
    (correctly) refused to run."""
    monkeypatch.setattr(S.os, "cpu_count", lambda: cores)
    assert S._default_worker_count(reserve=4) >= 1
    assert S._default_worker_count(reserve=2) >= 1


def test_default_worker_count_ignores_a_negative_reserve(monkeypatch):
    monkeypatch.setattr(S.os, "cpu_count", lambda: 8)
    assert S._default_worker_count(reserve=-4) == 8


def test_the_shipped_worker_defaults_are_usable_here():
    assert S.set_default_settings_preprocess_generate_masks({})["n_jobs"] >= 1
    assert S.get_measure_crop_settings({})["n_jobs"] >= 1


# ---------------------------------------------------------------------------
# _takes_an_argument — the call-shape decision behind defaults_for
# ---------------------------------------------------------------------------

def test_takes_an_argument_reads_the_signature():
    assert S._takes_an_argument(lambda settings: settings) is True
    assert S._takes_an_argument(lambda settings=None: settings) is True
    assert S._takes_an_argument(lambda *args: args) is True
    assert S._takes_an_argument(lambda: {}) is False
    assert S._takes_an_argument(lambda *, settings=None: settings) is False
    assert S._takes_an_argument(lambda **kw: kw) is False


def test_takes_an_argument_assumes_the_common_shape_when_uninspectable():
    """`print` is a builtin with no readable signature. The fallback must be
    the shape almost every defaults factory has, not a crash."""
    with pytest.raises(ValueError):
        inspect.signature(print)          # the premise of the test
    assert S._takes_an_argument(print) is True


def test_defaults_for_calls_a_zero_arg_factory_without_arguments(
        defaults_sandbox_local):
    S.register_defaults("zero_arg", lambda: {"a": 1})
    assert S.defaults_for("zero_arg", {"ignored": 2}) == {"a": 1}


# ---------------------------------------------------------------------------
# the registration seam (branches the existing seam tests leave)
# ---------------------------------------------------------------------------

@pytest.fixture
def defaults_sandbox_local():
    """Restore the registry and the four shared declaration tables.

    The category lists are restored IN PLACE rather than replaced with
    copies. Several of them are the same list object as a module-level name
    (`categories["Timelapse"] is timelapse_settings`), and
    tests/test_settings_categories.py asserts exactly that identity -- so a
    sandbox that put a copy back left the tables looking right while
    silently breaking the aliasing, and failed a later test in a different
    file.
    """
    registry = dict(S._DEFAULTS_REGISTRY)
    types_ = dict(S.expected_types)
    tips = dict(S.tooltips)
    cats = {name: (keys, list(keys)) for name, keys in S.categories.items()}
    descs = dict(S.descriptions)
    yield
    S._DEFAULTS_REGISTRY.clear()
    S._DEFAULTS_REGISTRY.update(registry)
    S.expected_types.clear()
    S.expected_types.update(types_)
    S.tooltips.clear()
    S.tooltips.update(tips)
    S.categories.clear()
    for name, (original_list, contents) in cats.items():
        original_list[:] = contents
        S.categories[name] = original_list
    S.descriptions.clear()
    S.descriptions.update(descs)


def test_register_defaults_rejects_a_blank_app_key(defaults_sandbox_local):
    with pytest.raises(ValueError) as exc:
        S.register_defaults("", lambda s: s)
    assert "need an app key" in str(exc.value)


def test_register_defaults_rejects_a_non_callable(defaults_sandbox_local):
    with pytest.raises(TypeError) as exc:
        S.register_defaults("notfn", {"a": 1})
    assert "are not callable" in str(exc.value)


def test_unregister_and_has_registered_round_trip(defaults_sandbox_local):
    assert S.has_registered_defaults("gone") is False
    assert S.unregister_defaults("gone") is False
    S.register_defaults("gone", lambda s: s)
    assert S.has_registered_defaults("gone") is True
    assert "gone" in S.registered_default_apps()
    assert S.unregister_defaults("gone") is True
    assert S.has_registered_defaults("gone") is False


def test_defaults_for_names_the_known_keys_when_asked_for_an_unknown_one():
    with pytest.raises(KeyError) as exc:
        S.defaults_for("no_such_module_key_at_all")
    assert "no defaults registered" in str(exc.value)


def test_defaults_for_rejects_a_factory_that_does_not_return_a_dict(
        defaults_sandbox_local):
    S.register_defaults("bad_return", lambda s: ["not", "a", "dict"])
    with pytest.raises(TypeError) as exc:
        S.defaults_for("bad_return")
    assert "returned list" in str(exc.value)


def test_defaults_for_hands_out_a_fresh_dict_each_time(defaults_sandbox_local):
    shared = {"a": 1}
    S.register_defaults("shared", lambda s: shared)
    first = S.defaults_for("shared")
    first["a"] = 999
    assert S.defaults_for("shared")["a"] == 1


# ---------------------------------------------------------------------------
# get_timelapse_settings / *settings=None* entry points
# ---------------------------------------------------------------------------

def test_get_timelapse_settings_forces_the_flag_the_module_is_named_for():
    # Even when the caller explicitly said False: `timelapse` is what the
    # Timelapse module IS, not a knob inside it.
    assert S.get_timelapse_settings({"timelapse": False})["timelapse"] is True
    assert S.get_timelapse_settings(None)["timelapse"] is True


def test_get_timelapse_settings_is_the_mask_defaults_plus_the_flag():
    mask = S.set_default_settings_preprocess_generate_masks({})
    tl = S.get_timelapse_settings(None)
    assert set(mask).issubset(set(tl))
    assert tl["timelapse_mode"] == mask["timelapse_mode"]


@pytest.mark.parametrize("fn_name", [
    "set_default_settings_preprocess_generate_masks",
    "get_timelapse_settings",
    "set_default_umap_image_settings",
    "get_measure_crop_settings",
    "set_default_generate_barecode_mapping",
    "set_default_stitch",
    "set_default_multichannel",
    "set_default_general",
    "get_automated_motility_assay_default_settings",
])
def test_settings_none_is_the_same_as_an_empty_dict(fn_name):
    """Every `settings=None` entry point exists so a caller can ask for the
    defaults with no dict at all. It must produce exactly the empty-dict
    answer, not a shorter one."""
    fn = getattr(S, fn_name)
    assert fn(None) == fn({})


# ---------------------------------------------------------------------------
# Cellpose model discovery
# ---------------------------------------------------------------------------

@pytest.fixture
def cellpose_cache_reset():
    before = S._CELLPOSE_MODELS_CACHE
    S._CELLPOSE_MODELS_CACHE = None
    yield
    S._CELLPOSE_MODELS_CACHE = before


def _fake_cellpose(monkeypatch, model_names, user_models=None, raises=False):
    """Install a stand-in `cellpose.models` and make it look already-imported."""
    cellpose = types.ModuleType("cellpose")
    models = types.ModuleType("cellpose.models")
    models.MODEL_NAMES = model_names

    def _get_user_models():
        if raises:
            raise OSError("gui_models.txt is a directory")
        return user_models or []

    models.get_user_models = _get_user_models
    cellpose.models = models
    monkeypatch.setitem(sys.modules, "cellpose", cellpose)
    monkeypatch.setitem(sys.modules, "cellpose.models", models)
    return models


def test_read_cellpose_models_returns_nothing_when_cellpose_is_absent(
        monkeypatch):
    real_import = __import__

    def _fake_import(name, *a, **kw):
        if name == "cellpose" or name.startswith("cellpose."):
            raise ImportError("no cellpose here")
        return real_import(name, *a, **kw)

    monkeypatch.delitem(sys.modules, "cellpose", raising=False)
    monkeypatch.delitem(sys.modules, "cellpose.models", raising=False)
    monkeypatch.setattr("builtins.__import__", _fake_import)
    assert S._read_cellpose_models() == ()


def test_read_cellpose_models_merges_stock_and_user_registered(monkeypatch):
    _fake_cellpose(monkeypatch, ["cpsam"], ["my_cells", "my_nuclei"])
    assert S._read_cellpose_models() == ("cpsam", "my_cells", "my_nuclei")


def test_read_cellpose_models_keeps_the_stock_list_when_the_registry_is_broken(
        monkeypatch):
    """A malformed ~/.cellpose/models/gui_models.txt must not cost the user
    the models that DO exist."""
    _fake_cellpose(monkeypatch, ["cpsam"], raises=True)
    assert S._read_cellpose_models() == ("cpsam",)


def test_read_cellpose_models_strips_dedupes_and_drops_blanks(monkeypatch):
    _fake_cellpose(monkeypatch, [" cpsam ", "cpsam", ""], ["  ", "mine",
                                                           "mine"])
    assert S._read_cellpose_models() == ("cpsam", "mine")


def test_read_cellpose_models_tolerates_a_missing_MODEL_NAMES(monkeypatch):
    models = _fake_cellpose(monkeypatch, ["cpsam"], ["mine"])
    del models.MODEL_NAMES
    assert S._read_cellpose_models() == ("mine",)


def test_read_cellpose_models_tolerates_a_None_MODEL_NAMES(monkeypatch):
    _fake_cellpose(monkeypatch, None, ["mine"])
    assert S._read_cellpose_models() == ("mine",)


def test_cellpose_model_choices_does_not_import_cellpose_by_default(
        monkeypatch, cellpose_cache_reset):
    """Importing cellpose costs ~2.5s and this runs while a panel is built."""
    monkeypatch.delitem(sys.modules, "cellpose.models", raising=False)
    called = []
    monkeypatch.setattr(S, "_read_cellpose_models",
                        lambda: called.append(1) or ("live",))
    assert S.cellpose_model_choices() == tuple(S.CELLPOSE_MODEL_CHOICES)
    assert called == []
    # ...and a miss must NOT be cached, or the fallback is pinned for the
    # whole process even after Cellpose loads.
    assert S._CELLPOSE_MODELS_CACHE is None


def test_cellpose_model_choices_asks_when_told_to_block(
        monkeypatch, cellpose_cache_reset):
    monkeypatch.delitem(sys.modules, "cellpose.models", raising=False)
    monkeypatch.setattr(S, "_read_cellpose_models", lambda: ("mine", "cpsam"))
    got = S.cellpose_model_choices(block=True)
    assert got == ("cpsam", "mine")           # cpsam first
    assert S._CELLPOSE_MODELS_CACHE == got    # a real answer IS cached


def test_cellpose_model_choices_reads_when_cellpose_is_already_imported(
        monkeypatch, cellpose_cache_reset):
    _fake_cellpose(monkeypatch, ["cpsam"], ["trained_on_my_plate"])
    assert S.cellpose_model_choices() == ("cpsam", "trained_on_my_plate")


def test_cellpose_model_choices_serves_the_cache_and_refresh_busts_it(
        monkeypatch, cellpose_cache_reset):
    answers = [("cpsam", "first"), ("cpsam", "second")]
    monkeypatch.setattr(S, "_read_cellpose_models", lambda: answers.pop(0))
    first = S.cellpose_model_choices(block=True)
    assert S.cellpose_model_choices(block=True) is first    # cached
    assert S.cellpose_model_choices(block=True, refresh=True) == (
        "cpsam", "second")


def test_cellpose_model_choices_falls_back_when_the_api_says_nothing(
        monkeypatch, cellpose_cache_reset):
    monkeypatch.setattr(S, "_read_cellpose_models", lambda: ())
    assert S.cellpose_model_choices(block=True) == tuple(
        S.CELLPOSE_MODEL_CHOICES)
    assert S._CELLPOSE_MODELS_CACHE is None


def test_cpsam_first_moves_the_default_to_the_front():
    assert S._cpsam_first(["a", "cpsam", "b"]) == ("cpsam", "a", "b")


def test_cpsam_first_leaves_a_list_without_the_default_alone():
    # A Cellpose that has no cpsam is not one spaCR should reorder.
    assert S._cpsam_first(["a", "b"]) == ("a", "b")


def test_cellpose_model_menu_offers_the_legacy_spellings_too(
        monkeypatch, cellpose_cache_reset):
    monkeypatch.setattr(S, "_read_cellpose_models", lambda: ("cpsam", "mine"))
    menu = S.cellpose_model_menu(block=True)
    assert menu[:2] == ("cpsam", "mine")
    # A user whose saved settings say 'cyto2' must SEE their own value in the
    # combo rather than have it silently replaced.
    for legacy in S._CELLPOSE_ALIASES:
        assert legacy in menu
    assert len(menu) == len(set(menu))


def test_cellpose_model_menu_does_not_duplicate_a_live_alias(
        monkeypatch, cellpose_cache_reset):
    monkeypatch.setattr(S, "_read_cellpose_models",
                        lambda: ("cpsam", "cyto2"))
    menu = S.cellpose_model_menu(block=True)
    assert menu.count("cyto2") == 1


# ---------------------------------------------------------------------------
# normalize_cellpose_model_name
# ---------------------------------------------------------------------------

@pytest.fixture
def cellpose_notices_reset():
    from spacr import utils as U
    before = set(U._REPORTED_CELLPOSE_NOTICES)
    U._REPORTED_CELLPOSE_NOTICES.clear()
    yield
    U._REPORTED_CELLPOSE_NOTICES.clear()
    U._REPORTED_CELLPOSE_NOTICES.update(before)


@pytest.mark.parametrize("value", [None, "", "   "])
def test_normalize_cellpose_model_name_fills_an_empty_setting(
        value, cellpose_notices_reset):
    assert S.normalize_cellpose_model_name(value) == "cpsam"


def test_normalize_cellpose_model_name_maps_every_legacy_spelling(
        cellpose_notices_reset, capsys):
    from spacr.utils import LEGACY_CELLPOSE_MODELS
    for legacy in LEGACY_CELLPOSE_MODELS:
        assert S.normalize_cellpose_model_name(legacy) == "cpsam"
    out = capsys.readouterr().out
    assert out.count("predates Cellpose-SAM") == len(LEGACY_CELLPOSE_MODELS)


def test_normalize_cellpose_model_name_names_the_object_and_the_key(
        cellpose_notices_reset, capsys):
    S.normalize_cellpose_model_name("cyto2", object_type="nucleus",
                                    key="nucleus_model_name")
    out = capsys.readouterr().out
    assert "(nucleus_model_name)" in out
    assert "for nucleus" in out


def test_normalize_cellpose_model_name_says_it_once_per_run(
        cellpose_notices_reset, capsys):
    for _ in range(5):
        S.normalize_cellpose_model_name("cyto3", object_type="cell",
                                        key="cell_model_name")
    # One line, not one per field: a 1000-field plate used to bury the log.
    assert capsys.readouterr().out.count("predates Cellpose-SAM") == 1


def test_normalize_cellpose_model_name_passes_a_checkpoint_through(
        cellpose_notices_reset, capsys):
    path = "/models/my_cells.pth"
    assert S.normalize_cellpose_model_name(path) == path
    assert S.normalize_cellpose_model_name("  cpsam  ") == "cpsam"
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# _get_object_settings
# ---------------------------------------------------------------------------

def _mask_settings(**over):
    s = S.set_default_settings_preprocess_generate_masks({})
    s.update(over)
    return s


@pytest.mark.parametrize("obj", ["cell", "nucleus", "pathogen"])
def test_get_object_settings_derives_the_size_window_from_the_diameter(obj):
    s = _mask_settings(**{f"{obj}_diameter": "40.0", "verbose": False})
    got = S._get_object_settings(obj, s)
    # A CSV-imported diameter arrives as a STRING; not coercing it made the
    # size window a string-multiplication TypeError.
    assert got["diameter"] == 40.0
    assert got["minimum_size"] == 40.0 ** 2 / 4
    assert got["maximum_size"] == 40.0 ** 2 * 10
    assert got["model_name"] == "cpsam"
    assert got["min_size"] == s[f"{obj}_min_area"]


@pytest.mark.parametrize("obj,label", [("cell", "Cell"),
                                       ("nucleus", "Nucleus"),
                                       ("pathogen", "Pathogen")])
def test_get_object_settings_keeps_the_magnification_diameter_on_bad_input(
        obj, label, capsys):
    s = _mask_settings(**{f"{obj}_diameter": "thirty", "verbose": False})
    baseline = S._get_object_settings(obj, _mask_settings(verbose=False))
    got = S._get_object_settings(obj, s)
    # It falls back to the magnification-derived diameter and SAYS so, rather
    # than segmenting with a diameter of NaN.
    assert got["diameter"] == baseline["diameter"]
    assert f"{label} diameter must be an integer or float" in capsys.readouterr().out


def test_get_object_settings_uses_the_magnification_when_no_diameter_is_set():
    s = _mask_settings(magnification=20, cell_diameter=None, verbose=False)
    from spacr.utils import _get_diam
    assert S._get_object_settings("cell", s)["diameter"] == _get_diam(
        20, obj="cell")


def test_get_object_settings_pathogen_does_not_resample_and_can_merge():
    s = _mask_settings(merge_pathogens=True, verbose=False)
    got = S._get_object_settings("pathogen", s)
    assert got["resample"] is False      # pathogens only
    assert got["merge"] is True
    assert S._get_object_settings("cell", s)["resample"] is True
    assert S._get_object_settings("cell", s)["merge"] is False


def test_get_object_settings_carries_the_restore_type_through():
    s = _mask_settings(cell_restore_type="denoise", verbose=False)
    assert S._get_object_settings("cell", s)["restore_type"] == "denoise"
    assert S._get_object_settings(
        "nucleus", _mask_settings(verbose=False))["restore_type"] is None


def test_get_object_settings_maps_a_legacy_model_name_forward(
        cellpose_notices_reset):
    s = _mask_settings(cell_model_name="cyto2", verbose=False)
    assert S._get_object_settings("cell", s)["model_name"] == "cpsam"


def test_get_object_settings_says_so_for_an_object_it_cannot_configure(capsys):
    """'cell_large' is the one object type that reaches the else arm.

    Everything else -- 'cytoplasm', 'organelle', a typo -- is rejected by
    _get_diam on the function's first line, so it never gets here.
    'cell_large' is sized by _get_diam but has no per-object branch, so it
    comes back WITHOUT min_size / filter_size / restore_type, which the
    segmentation caller then indexes.
    """
    from spacr.utils import _get_diam
    s = _mask_settings(verbose=False)
    got = S._get_object_settings("cell_large", s)
    out = capsys.readouterr().out
    assert "Object type: cell_large not supported" in out
    assert got["diameter"] == _get_diam(s["magnification"], obj="cell_large")
    assert "min_size" not in got and "restore_type" not in got


@pytest.mark.parametrize("obj", ["cytoplasm", "organelle", "celll"])
def test_get_object_settings_stops_on_an_object_type_it_cannot_size(obj):
    with pytest.raises(ValueError) as exc:
        S._get_object_settings(obj, _mask_settings(verbose=False))
    assert "unsupported object type" in str(exc.value)


def test_get_object_settings_verbose_prints_the_dict(capsys):
    S._get_object_settings("cell", _mask_settings(verbose=True))
    assert "'diameter'" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# get_measure_crop_settings
# ---------------------------------------------------------------------------

def test_measure_crop_settings_parses_bracketed_strings_from_a_csv_import():
    """The Qt drag-and-drop import reads CSV cells as raw strings and does not
    run them through check_settings; without this measure_crop rejects them."""
    got = S.get_measure_crop_settings({
        "channels": "[0, 1, 2]",
        "png_size": "(224, 224)",
        "crop_mode": "['cell', 'nucleus']",
    })
    assert got["channels"] == [0, 1, 2]
    assert got["png_size"] == (224, 224)
    assert got["crop_mode"] == ["cell", "nucleus"]


def test_measure_crop_settings_leaves_an_unparseable_bracket_string_alone():
    got = S.get_measure_crop_settings({"experiment": "[not python"})
    assert got["experiment"] == "[not python"


def test_measure_crop_settings_leaves_ordinary_values_untouched():
    got = S.get_measure_crop_settings({"channels": [3], "experiment": "exp7"})
    assert got["channels"] == [3]
    assert got["experiment"] == "exp7"


def test_measure_crop_settings_test_mode_turns_on_verbose_and_plot(capsys):
    got = S.get_measure_crop_settings({"test_mode": True, "test_nr": 7})
    assert got["verbose"] is True and got["plot"] is True
    assert "Test mode enabled with 7 images" in capsys.readouterr().out


def test_measure_crop_settings_leaves_plot_off_when_not_testing():
    got = S.get_measure_crop_settings({})
    assert got["test_mode"] is False
    assert got["plot"] is False and got["verbose"] is False


def test_measure_crop_settings_defaults_the_organelle_rollup():
    """The omission that made a measure run write no organelle table at all.

    measure.py gates all four organelle writes on `is not None`, and this key
    was only ever defaulted in the MASK pipeline's factory — which never
    reaches the measure settings.
    """
    assert S.get_measure_crop_settings({})["summarize_organelles_by"] == "cell"


def test_measure_crop_settings_respects_an_explicit_organelle_rollup():
    got = S.get_measure_crop_settings({"summarize_organelles_by": None})
    assert got["summarize_organelles_by"] is None


# ---------------------------------------------------------------------------
# get_perform_regression_default_settings — quantile validation
# ---------------------------------------------------------------------------

def _regression(**over):
    s = {"regression_type": "quantile"}
    s.update(over)
    return s


def test_quantile_regression_refuses_a_leftover_alpha():
    """alpha used to double as the quantile. A CSV reading alpha=0.9 meant two
    different things under two regression types; dropping it silently would
    fit the median and label the output 0.9."""
    with pytest.raises(ValueError) as exc:
        S.get_perform_regression_default_settings(_regression(alpha=0.9))
    msg = str(exc.value)
    assert "does not use alpha" in msg
    assert "quantile=0.9 instead" in msg


@pytest.mark.parametrize("bad", [0, 1, -0.5, 1.5, "0.9", True, None])
def test_quantile_regression_refuses_a_quantile_outside_the_open_interval(bad):
    with pytest.raises(ValueError) as exc:
        S.get_perform_regression_default_settings(_regression(quantile=bad))
    assert "strictly inside (0, 1)" in str(exc.value)


def test_quantile_regression_drops_well_aggregation(capsys):
    """The quantile of a per-well average is not the quantile of the
    response."""
    got = S.get_perform_regression_default_settings(
        _regression(quantile=0.9, agg_type="mean"))
    assert got["agg_type"] is None
    out = capsys.readouterr().out
    assert "Fitting the 0.9 quantile" in out
    assert "agg_type set to None" in out


def test_non_quantile_regression_keeps_alpha_and_aggregation():
    got = S.get_perform_regression_default_settings(
        {"regression_type": "ridge", "alpha": 0.9, "agg_type": "mean"})
    assert got["alpha"] == 0.9
    assert got["agg_type"] == "mean"


def test_regression_defaults_track_the_dependent_variable():
    got = S.get_perform_regression_default_settings(
        {"dependent_variable": "pathogen_area"})
    # score_column must be the column being regressed, or the simulated
    # minimum cell count describes a different measurement.
    assert got["score_column"] == "pathogen_area"


def test_regression_control_wells_default_from_filter_value():
    assert S.get_perform_regression_default_settings(
        {"filter_value": ["c1", "c2"]})["control_wells"] == ["c1", "c2"]
    # filter_value is indexed and iterated, so None is not a legal value here.
    assert S.get_perform_regression_default_settings(
        {"filter_value": None})["control_wells"] == []


# ---------------------------------------------------------------------------
# check_settings — every type branch, and the answer it produces
# ---------------------------------------------------------------------------

class _Var:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


def _vd(**kv):
    return {k: ("label", None, _Var(v), None) for k, v in kv.items()}


class _RecordingQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def test_check_settings_forwards_every_error_to_the_queue():
    q = _RecordingQueue()
    settings, errors = S.check_settings(
        _vd(unknown_key_xyz="1"), {}, q=q)
    assert errors and q.items == errors


def test_check_settings_makes_its_own_queue_when_not_given_one():
    settings, errors = S.check_settings(_vd(src="/data"), {"src": str})
    assert settings["src"] == "/data"
    assert errors == []


def test_check_settings_accepts_a_category_name_as_a_key():
    # category_keys are widgets too; they must not be reported as unknown.
    key = S.category_keys[0]
    settings, errors = S.check_settings(_vd(**{key: "x"}), {})
    assert errors == []
    assert settings[key] == "x"


# --- the list-of-lists keys -------------------------------------------------

# The literal-eval'd keys are matched by NAME, so they need to be declared
# (or be a category name) to get past the unknown-key guard first.
_LIST_KEY_TYPES = {"timelapse_objects": list, "cell_plate_metadata": list,
                   "png_size": list, "crop_mode": list}


def test_check_settings_list_key_accepts_a_flat_list():
    settings, errors = S.check_settings(
        _vd(timelapse_objects="['cell', 'nucleus']"), _LIST_KEY_TYPES)
    assert settings["timelapse_objects"] == ["cell", "nucleus"]
    assert errors == []


def test_check_settings_list_key_accepts_a_list_of_lists():
    settings, _ = S.check_settings(
        _vd(cell_plate_metadata="[['c1'], ['c2']]"), _LIST_KEY_TYPES)
    # These keys bypass the plain `list` branch precisely so the nested form
    # survives; parse_list would reject it as "mixed types".
    assert settings["cell_plate_metadata"] == [["c1"], ["c2"]]


@pytest.mark.parametrize("blank", ["None", ""])
def test_check_settings_list_key_accepts_a_blank_as_none(blank):
    """Blank means "not set", and for these keys that is a shipped default.

    It used to raise "Expected a list ... but got NoneType": the `value is
    None` arm assigned parsed_value = None and then fell into the very next
    `else`, so no value could reach the list branch through it.
    """
    settings, errors = S.check_settings(
        _vd(cell_plate_metadata=blank), _LIST_KEY_TYPES)
    assert settings["cell_plate_metadata"] is None
    assert errors == []


@pytest.mark.parametrize("fn_name", [
    "get_analyze_recruitment_default_settings",
    "set_analyze_class_proportion_defaults",
    "set_analyze_endodyogeny_defaults",
    "set_analyze_invasion_defaults",
    "set_analyze_replication_defaults",
    "set_default_plot_data_from_db",
])
def test_the_plate_metadata_defaults_survive_the_validator(fn_name):
    """Each of these ships None for one or more *_plate_metadata key.

    Running the module from the Tk panel without touching anything used to
    report an error per key and return a dict missing all of them.
    """
    defaults = getattr(S, fn_name)({})
    metadata_keys = [k for k in defaults if k.endswith("_plate_metadata")]
    assert metadata_keys, f"{fn_name} no longer has plate metadata keys"
    vd = _vd(**{k: "None" for k in metadata_keys})
    settings, errors = S.check_settings(vd, S.expected_types)
    assert errors == [], errors
    for key in metadata_keys:
        assert key in settings and settings[key] is None


def test_check_settings_list_key_does_not_unwrap_a_single_element():
    # Unlike the plain `list` branch: [224] stays a list here.
    settings, _ = S.check_settings(_vd(png_size="[224]"), _LIST_KEY_TYPES)
    assert settings["png_size"] == [224]


def test_check_settings_list_key_rejects_mixed_depth():
    settings, errors = S.check_settings(
        _vd(cell_plate_metadata="['c1', ['c2']]"), _LIST_KEY_TYPES)
    # A half-nested value is ambiguous; the run must not guess which it meant.
    assert "cell_plate_metadata" not in settings
    assert any("mixed types" in e for e in errors)


def test_check_settings_list_key_rejects_a_scalar():
    settings, errors = S.check_settings(_vd(png_size="224"), _LIST_KEY_TYPES)
    assert any("Expected a list for 'png_size'" in e for e in errors)
    assert any("but got int" in e for e in errors)


def test_check_settings_list_key_rejects_unparseable_text():
    settings, errors = S.check_settings(
        _vd(crop_mode="cell,nucleus"), _LIST_KEY_TYPES)
    assert any("invalid format" in e for e in errors)


# --- list / bool / numeric --------------------------------------------------

def test_check_settings_list_type_unwraps_a_single_element():
    settings, _ = S.check_settings(_vd(channels="[2]"), {"channels": list})
    # A one-element list becomes the scalar: `channels=[2]` and `channels=2`
    # are the same request downstream.
    assert settings["channels"] == 2
    settings, _ = S.check_settings(_vd(channels="[0,1]"), {"channels": list})
    assert settings["channels"] == [0, 1]


def test_check_settings_list_type_maps_empty_to_none():
    settings, _ = S.check_settings(_vd(channels=""), {"channels": list})
    assert settings["channels"] is None


@pytest.mark.parametrize("raw,want", [
    ("True", True), ("true", True), ("1", True), ("t", True), ("y", True),
    ("yes", True), ("False", False), ("no", False), ("maybe", False),
])
def test_check_settings_bool_reads_the_usual_spellings(raw, want):
    settings, _ = S.check_settings(_vd(plot=raw), {"plot": bool})
    assert settings["plot"] is want


def test_check_settings_bool_of_a_blank_is_false():
    # '' becomes None at the top, and bool(None) is False.
    settings, _ = S.check_settings(_vd(plot=""), {"plot": bool})
    assert settings["plot"] is False


def test_check_settings_int_or_none():
    exp = {"n": (int, type(None))}
    assert S.check_settings(_vd(n="5"), exp)[0]["n"] == 5
    assert S.check_settings(_vd(n="None"), exp)[0]["n"] is None
    settings, errors = S.check_settings(_vd(n="5.5"), exp)
    assert any("Expected an integer or None" in e for e in errors)


def test_check_settings_float_or_none():
    exp = {"f": (float, type(None))}
    assert S.check_settings(_vd(f="0.25"), exp)[0]["f"] == 0.25
    assert S.check_settings(_vd(f=""), exp)[0]["f"] is None
    settings, errors = S.check_settings(_vd(f="1e-3"), exp)
    assert any("Expected a float or None" in e for e in errors)


def test_check_settings_int_or_float_keeps_the_written_type():
    exp = {"v": (int, float)}
    assert S.check_settings(_vd(v="7"), exp)[0]["v"] == 7
    assert isinstance(S.check_settings(_vd(v="7"), exp)[0]["v"], int)
    assert S.check_settings(_vd(v="7.5"), exp)[0]["v"] == 7.5
    settings, errors = S.check_settings(_vd(v="seven"), exp)
    assert any("Expected an integer or float" in e for e in errors)


# --- (bool, int): the one that silently inverted every score ----------------

@pytest.mark.parametrize("raw,want", [
    ("True", True), ("t", True), ("yes", True),
    ("False", False), ("f", False), ("no", False),
    ("0", 0), ("1", 1), ("-1", -1),
])
def test_check_settings_bool_int_keeps_false_false_and_minus_one_minus_one(
        raw, want):
    """invert_dependent_variable: False/0 = as measured, True/1 = 1-x,
    -1 = 1/x. The generic tuple branch reaches bool('False') first, which is
    True, and would invert every score in the screen."""
    exp = {"invert_dependent_variable": (bool, int)}
    settings, _ = S.check_settings(
        _vd(invert_dependent_variable=raw), exp)
    got = settings["invert_dependent_variable"]
    assert got == want and isinstance(got, type(want))


def test_check_settings_bool_int_accepts_none():
    exp = {"invert_dependent_variable": (bool, int)}
    settings, _ = S.check_settings(_vd(invert_dependent_variable="None"), exp)
    assert settings["invert_dependent_variable"] is None


def test_check_settings_bool_int_rejects_anything_else():
    exp = {"invert_dependent_variable": (bool, int)}
    settings, errors = S.check_settings(
        _vd(invert_dependent_variable="sometimes"), exp)
    assert any("Expected True, False or an integer" in e for e in errors)


# --- (list, None): y_lims and the broken axis -------------------------------

def test_check_settings_list_or_none_keeps_the_nested_broken_axis_form():
    """The generic tuple branch reaches list('[0, 5]') first and hands the
    pipeline ['[', '0', ',', ' ', '5', ']']."""
    exp = {"y_lims": (list, type(None))}
    assert S.check_settings(_vd(y_lims="[0, 5]"), exp)[0]["y_lims"] == [0, 5]
    assert S.check_settings(
        _vd(y_lims="[[0, 1], [8, 9]]"), exp)[0]["y_lims"] == [[0, 1], [8, 9]]


def test_check_settings_list_or_none_accepts_none_and_a_tuple():
    exp = {"y_lims": (list, type(None))}
    assert S.check_settings(_vd(y_lims="None"), exp)[0]["y_lims"] is None
    assert S.check_settings(_vd(y_lims="(0, 5)"), exp)[0]["y_lims"] == [0, 5]


def test_check_settings_list_or_none_passes_a_real_list_through():
    exp = {"y_lims": (list, type(None))}
    assert S.check_settings(_vd(y_lims=[1, 2]), exp)[0]["y_lims"] == [1, 2]


def test_check_settings_list_or_none_rejects_a_scalar_and_bad_text():
    exp = {"y_lims": (list, type(None))}
    _, errors = S.check_settings(_vd(y_lims="5"), exp)
    assert any("but got int" in e for e in errors)
    _, errors = S.check_settings(_vd(y_lims="0, 5]"), exp)
    assert any("Expected a list or None" in e for e in errors)


# --- str-ish tuples ---------------------------------------------------------

def test_check_settings_str_or_none():
    exp = {"custom_regex": (str, type(None))}
    assert S.check_settings(_vd(custom_regex="r.*"), exp)[0]["custom_regex"] == "r.*"
    assert S.check_settings(_vd(custom_regex="None"), exp)[0]["custom_regex"] is None


def test_check_settings_str_none_or_list_handles_all_three():
    exp = {"file_metadata": (str, type(None), list)}
    assert S.check_settings(
        _vd(file_metadata="/a"), exp)[0]["file_metadata"] == "/a"
    assert S.check_settings(
        _vd(file_metadata="None"), exp)[0]["file_metadata"] is None
    # A value that is already the declared list type has to survive. This
    # branch called parse_list, which literal_eval()s its argument and so
    # raised on a real list -- the value was dropped and a format error
    # logged for a value that was already correct.
    got, errors = S.check_settings(_vd(file_metadata=["/a", "/b"]), exp)
    assert got["file_metadata"] == ["/a", "/b"]
    assert errors == []
    # A non-str, non-list, non-None value has no reading: it becomes None
    # rather than str(42).
    assert S.check_settings(
        _vd(file_metadata=42), exp)[0]["file_metadata"] is None


def test_check_settings_str_none_or_list_maps_an_empty_list_to_none():
    exp = {"file_metadata": (str, type(None), list)}
    assert S.check_settings(
        _vd(file_metadata=[]), exp)[0]["file_metadata"] is None


# --- dict -------------------------------------------------------------------

def test_check_settings_dict_parses_and_reports():
    exp = {"d": dict}
    assert S.check_settings(_vd(d="{'a': 1}"), exp)[0]["d"] == {"a": 1}
    settings, errors = S.check_settings(_vd(d="[1, 2]"), exp)
    assert settings["d"] == {}
    assert any("Expected a dictionary for 'd'" in e for e in errors)
    settings, errors = S.check_settings(_vd(d=17), exp)
    assert settings["d"] == {}
    assert any("string representation of a dictionary" in e for e in errors)


# --- the generic branches ---------------------------------------------------

def test_check_settings_generic_tuple_takes_the_first_type_that_works():
    exp = {"v": (int, str)}
    assert S.check_settings(_vd(v="12"), exp)[0]["v"] == 12
    assert S.check_settings(_vd(v="ab"), exp)[0]["v"] == "ab"


def test_check_settings_generic_tuple_reports_when_nothing_fits():
    exp = {"v": (int, float)}
    # (int, float) has its own branch, so use a tuple that reaches the generic
    # loop and cannot accept the value.
    exp = {"v": (int, complex)}
    settings, errors = S.check_settings(_vd(v="not a number"), exp)
    assert any("does not match any expected types" in e for e in errors)


def test_check_settings_generic_tuple_maps_empty_to_none():
    exp = {"v": (int, str)}
    assert S.check_settings(_vd(v=""), exp)[0]["v"] is None


def test_check_settings_plain_type_and_its_error():
    assert S.check_settings(_vd(n="42"), {"n": int})[0]["n"] == 42
    assert S.check_settings(_vd(n=""), {"n": int})[0]["n"] is None
    settings, errors = S.check_settings(_vd(n="forty"), {"n": int})
    assert any("Expected type int for 'n'" in e for e in errors)


def test_check_settings_untyped_key_defaults_to_str():
    key = S.category_keys[0]
    settings, errors = S.check_settings(_vd(**{key: "7"}), {})
    assert settings[key] == "7"


def test_check_settings_collects_every_error_rather_than_stopping():
    exp = {"a": int, "b": int, "c": int}
    settings, errors = S.check_settings(_vd(a="x", b="y", c="3"), exp)
    assert len(errors) == 2
    assert settings["c"] == 3      # the good value still got through


def test_check_settings_round_trips_the_shipped_mask_defaults():
    """The real oracle: every shipped default must survive the validator that
    the GUI runs over it, as the type it was declared with."""
    defaults = S.set_default_settings_preprocess_generate_masks({})
    vd = _vd(**{k: ("None" if v is None else str(v))
                for k, v in defaults.items()})
    settings, errors = S.check_settings(vd, S.expected_types)
    unknown = [e for e in errors if "not found in expected types" in e]
    assert not unknown, unknown
    assert not errors, errors
    assert settings["magnification"] == defaults["magnification"]
    assert settings["cell_channel"] is None
    assert settings["channels"] == defaults["channels"]


# ---------------------------------------------------------------------------
# generate_fields / generate_fields_lazy  (Tk)
# ---------------------------------------------------------------------------

class _Scrollable:
    """The `.scrollable_frame` shape the settings panel hands these two."""

    def __init__(self, frame):
        self.scrollable_frame = frame


@pytest.fixture
def scrollable(tk_root):
    import tkinter as tk
    frame = tk.Frame(tk_root)
    return _Scrollable(frame)


def test_generate_fields_builds_a_widget_per_setting(scrollable):
    variables = {
        "src": ("entry", None, "/data"),
        "plot": ("check", None, True),
        "metadata_type": ("combo", ["cellvoyager", "cq1"], "cq1"),
    }
    vars_dict = S.generate_fields(variables, scrollable)
    assert set(vars_dict) == set(variables)
    for key, (label, widget, var, frame) in vars_dict.items():
        assert widget is not None
    # The widget really holds the default it was given — that is what the
    # panel reads back through check_settings.
    assert vars_dict["src"][2].get() == "/data"
    assert vars_dict["plot"][2].get() is True
    assert vars_dict["metadata_type"][2].get() == "cq1"


def test_generate_fields_ticks_once_per_field(scrollable):
    ticks = []
    S.generate_fields({"src": ("entry", None, "/a"),
                       "plot": ("check", None, False)},
                      scrollable, tick_callback=lambda: ticks.append(1))
    assert len(ticks) == 2


def test_generate_fields_falls_back_to_a_type_default_on_a_bad_value(
        scrollable, monkeypatch, capsys):
    """A settings CSV can carry a value the widget cannot take. Losing the
    field entirely would silently drop the setting from the run."""
    calls = []
    # generate_fields imports create_input_field from .gui_utils inside the
    # function body, so patching the module attribute is what it will see.
    import spacr.gui_utils as GU
    original = GU.create_input_field

    def _picky(frame, key, row, var_type, options, default_value):
        calls.append(default_value)
        if default_value == "!!bad!!":
            raise ValueError("widget refuses this")
        return original(frame, key, row, var_type, options, default_value)

    monkeypatch.setattr(GU, "create_input_field", _picky)
    vars_dict = S.generate_fields({"src": ("entry", None, "!!bad!!")},
                                  scrollable)
    assert calls == ["!!bad!!", ""]          # retried with the type default
    assert vars_dict["src"][2].get() == ""
    assert "reverting to" in capsys.readouterr().out


@pytest.mark.parametrize("var_type,options,fallback", [
    ("check", None, False),
    ("entry", None, ""),
    ("combo", ["a", "b"], "a"),
    ("int", None, 0),
    ("float", None, 0.0),
])
def test_generate_fields_fallback_matches_the_widget_type(
        scrollable, monkeypatch, var_type, options, fallback):
    seen = []
    import spacr.gui_utils as GU
    original = GU.create_input_field

    def _picky(frame, key, row, vt, opts, default_value):
        seen.append(default_value)
        if len(seen) == 1:
            raise ValueError("no")
        return original(frame, key, row, vt, opts, default_value)

    monkeypatch.setattr(GU, "create_input_field", _picky)
    S.generate_fields({"k": (var_type, options, object())}, scrollable)
    assert seen[1] == fallback


def test_generate_fields_skips_a_field_that_fails_twice(scrollable,
                                                        monkeypatch, capsys):
    import spacr.gui_utils as GU

    def _always_fails(*a, **kw):
        raise ValueError("no widget for you")

    monkeypatch.setattr(GU, "create_input_field", _always_fails)
    vars_dict = S.generate_fields({"src": ("entry", None, "/a"),
                                   "plot": ("check", None, True)}, scrollable)
    assert vars_dict == {}
    out = capsys.readouterr().out
    assert out.count("Could not create field") == 2


def test_generate_fields_attaches_a_tooltip_where_one_exists(scrollable,
                                                             monkeypatch):
    attached = []
    import spacr.gui_elements as GE
    monkeypatch.setattr(GE, "spacrToolTip",
                        lambda widget, text: attached.append(text))
    key = next(k for k in S.tooltips if k)
    S.generate_fields({key: ("entry", None, ""),
                       "not_a_real_setting_key": ("entry", None, "")},
                      scrollable)
    assert attached == [S.tooltips[key]]


def test_generate_fields_lazy_defers_the_categorised_settings(scrollable):
    categorised = next(k for keys in S.categories.values() for k in keys)
    variables = {categorised: ("entry", None, "x"),
                 "not_a_real_setting_key": ("entry", None, "y")}
    vars_dict = S.generate_fields_lazy(variables, scrollable)
    # A categorised key is a placeholder: no widget is built until its
    # heading is expanded. That is the whole point of the lazy variant.
    assert vars_dict[categorised] is None
    assert vars_dict["not_a_real_setting_key"] is not None
    # The definitions and the next row are stashed for the deferred build.
    assert scrollable._field_variables == variables
    assert scrollable._next_row == 2


def test_generate_fields_lazy_ticks_only_for_what_it_built(scrollable):
    categorised = next(k for keys in S.categories.values() for k in keys)
    ticks = []
    S.generate_fields_lazy(
        {categorised: ("entry", None, "x"),
         "not_a_real_setting_key": ("entry", None, "y")},
        scrollable, tick_callback=lambda: ticks.append(1))
    assert len(ticks) == 1


def test_generate_fields_lazy_falls_back_and_then_skips(scrollable,
                                                        monkeypatch, capsys):
    import spacr.gui_utils as GU
    original = GU.create_input_field
    seen = []

    def _picky(frame, key, row, vt, opts, default_value):
        seen.append((key, default_value))
        if key == "bad_key_never_works":
            raise ValueError("no")
        if key == "recovers_key" and default_value == "!!bad!!":
            raise ValueError("no")
        return original(frame, key, row, vt, opts, default_value)

    monkeypatch.setattr(GU, "create_input_field", _picky)
    vars_dict = S.generate_fields_lazy(
        {"recovers_key": ("entry", None, "!!bad!!"),
         "bad_key_never_works": ("entry", None, "x")}, scrollable)
    assert vars_dict["recovers_key"][2].get() == ""
    assert "bad_key_never_works" not in vars_dict
    out = capsys.readouterr().out
    assert "Warning: Invalid value for recovers_key" in out
    assert "Could not create field for 'bad_key_never_works'" in out


def test_generate_fields_lazy_attaches_tooltips_to_what_it_renders(
        scrollable, monkeypatch):
    uncategorised = next(
        k for k in S.tooltips
        if not any(k in keys for keys in S.categories.values()))
    attached = []
    import spacr.gui_elements as GE
    monkeypatch.setattr(GE, "spacrToolTip",
                        lambda widget, text: attached.append(text))
    S.generate_fields_lazy({uncategorised: ("entry", None, "")}, scrollable)
    assert attached == [S.tooltips[uncategorised]]


# ---------------------------------------------------------------------------
# the barcode-mapping defaults: the shipped regex has to actually work
# ---------------------------------------------------------------------------

def test_barecode_mapping_defaults_ship_a_usable_regex():
    """The default regex named `column`/`row` once, while the read processors
    read match.group('columnID') — so the shipped default raised
    "IndexError: no such group" on the first read."""
    import re
    got = S.set_default_generate_barecode_mapping(None)
    groups = set(re.compile(got["regex"]).groupindex)
    assert {"columnID", "rowID", "grna"}.issubset(groups)


def test_barecode_mapping_defaults_point_at_the_bundled_references():
    got = S.set_default_generate_barecode_mapping({})
    for key, kind in (("column_csv", "column"), ("grna_csv", "grna"),
                      ("row_csv", "row")):
        assert got[key] == S.bundled_barcode_path(kind)
        assert os.path.isfile(got[key])
    assert got["mode"] == "paired"
    assert got["single_direction"] == "R1"


def test_barecode_mapping_defaults_are_what_the_pipeline_indexes():
    """generate_barecode_mapping indexes, not .get()s, every one of these."""
    got = S.set_default_generate_barecode_mapping({})
    for key in ("src", "regex", "target_sequence", "offset_start",
                "expected_end", "column_csv", "grna_csv", "row_csv",
                "save_h5", "comp_type", "comp_level", "chunk_size", "n_jobs",
                "mode", "single_direction", "test", "fill_na"):
        assert key in got, key


def test_barecode_mapping_defaults_do_not_overwrite_a_caller_value():
    got = S.set_default_generate_barecode_mapping({"mode": "single",
                                                   "chunk_size": 7})
    assert got["mode"] == "single" and got["chunk_size"] == 7


# ---------------------------------------------------------------------------
# the motility assay defaults
# ---------------------------------------------------------------------------

def test_motility_assay_carries_its_own_source_folder():
    """It is a module of its own now; it used to inherit `src` from the mask
    settings it was merged into."""
    got = S.get_automated_motility_assay_default_settings(None)
    assert got["src"] == "path"
    assert got["tracked_object"] == "cell"
    # Exposed so a large run can skip the QC plotting work.
    assert got["infection_intensity_qc_graphs"] is True


def test_motility_assay_defaults_do_not_overwrite_a_caller_value():
    got = S.get_automated_motility_assay_default_settings({"src": "/plate"})
    assert got["src"] == "/plate"


# ---------------------------------------------------------------------------
# the whole-file invariants that catch the *next* omission
# ---------------------------------------------------------------------------

def _defaults_factories():
    out = []
    for name, fn in inspect.getmembers(S, inspect.isfunction):
        if fn.__module__ != "spacr.settings":
            continue
        if name.startswith("get_") or name.startswith("set_"):
            if name in ("set_default_generate_barecode_mapping",):
                out.append(name)
            elif name.startswith(("get_default", "get_", "set_")):
                out.append(name)
    return sorted(set(out))


@pytest.mark.parametrize("fn_name", _defaults_factories())
def test_a_defaults_factory_never_overwrites_a_caller_value(fn_name):
    """Idempotence is the property that lets the GUI merge a saved CSV over
    the defaults. A factory that assigns instead of setdefault silently
    discards what the user asked for."""
    fn = getattr(S, fn_name)
    sig = inspect.signature(fn)
    if not sig.parameters:
        pytest.skip("no settings argument")
    try:
        first = fn({})
    except (ValueError, KeyError):
        pytest.skip("needs specific keys to be exercised; covered elsewhere")
    if not isinstance(first, dict):
        pytest.skip("does not return a dict")
    second = fn(dict(first))
    # Keys that the factory deliberately derives (documented in the source)
    derived = {"verbose", "plot", "agg_type", "timelapse"}
    for key, value in first.items():
        if key in derived:
            continue
        assert second[key] == value or (
            second[key] != second[key] and value != value), key


@pytest.mark.parametrize("fn_name", _defaults_factories())
def test_a_defaults_factory_returns_a_json_shaped_dict(fn_name):
    """Every default is written to a settings CSV and read back. A value that
    is not a literal cannot survive that round trip."""
    fn = getattr(S, fn_name)
    if not inspect.signature(fn).parameters:
        result = fn()
    else:
        try:
            result = fn({})
        except (ValueError, KeyError):
            pytest.skip("needs specific keys; covered elsewhere")
    if not isinstance(result, dict):
        pytest.skip("does not return a dict")
    for key, value in result.items():
        assert isinstance(key, str), key
        assert isinstance(
            value, (str, int, float, bool, list, tuple, dict, type(None))
        ), f"{fn_name}[{key}] is a {type(value).__name__}"


def test_no_dead_setting_is_still_being_defaulted():
    """DEAD_SETTINGS names keys that were renamed away. A factory that still
    fills one of them re-creates the key the rename removed."""
    offenders = {}
    for fn_name in _defaults_factories():
        fn = getattr(S, fn_name)
        try:
            result = fn({}) if inspect.signature(fn).parameters else fn()
        except (ValueError, KeyError):
            continue
        if not isinstance(result, dict):
            continue
        dead = set(result) & set(S.DEAD_SETTINGS)
        if dead:
            offenders[fn_name] = sorted(dead)
    assert offenders == {}, offenders


def test_dead_settings_replacements_all_exist_or_are_none():
    """A rename map that points at a key nothing defaults is a broken
    migration: the user is told to use a setting that does not exist."""
    everything = set()
    for fn_name in _defaults_factories():
        fn = getattr(S, fn_name)
        try:
            result = fn({}) if inspect.signature(fn).parameters else fn()
        except (ValueError, KeyError):
            continue
        if isinstance(result, dict):
            everything |= set(result)
    missing = sorted(
        new for new in S.DEAD_SETTINGS.values()
        if new is not None and new not in everything and
        new not in S.expected_types)
    assert missing == [], missing


# ---------------------------------------------------------------------------
# The invariant that catches the NEXT undeclared key
# ---------------------------------------------------------------------------
#
# The panel builds its widget map from a module's own defaults factory, so any
# key that factory ships becomes a field. check_settings answers a field it
# cannot type with a "not found in expected types" warning and drops it, and
# gui_core.import_settings then does `if len(errors) > 0: return` -- pressing
# Run does nothing at all, with the reason only in the log queue. That is how
# Map Barcodes came to be unstartable from the Tk GUI on eleven keys at once.

#: The factories gui_core.setup_settings_panel dispatches, by module id. The
#: widget map -- and therefore what check_settings is asked to type -- is
#: exactly the keys these return. `categories` does NOT create a widget: it
#: only groups keys that a factory already shipped, which is why 'highlight'
#: and 'plate' can sit there, documented as inert, without costing anything.
_PANEL_FACTORIES = {
    "mask": "set_default_settings_preprocess_generate_masks",
    "measure": "get_measure_crop_settings",
    "classify": "deep_spacr_defaults",
    "umap": "set_default_umap_image_settings",
    "train_cellpose": "get_train_cellpose_default_settings",
    "ml_analyze": "set_default_analyze_screen",
    "cellpose_masks": "get_identify_masks_finetune_default_settings",
    "cellpose_all": "get_check_cellpose_models_default_settings",
    "map_barcodes": "set_default_generate_barecode_mapping",
    "recruitment": "get_analyze_recruitment_default_settings",
    "activation": "get_default_generate_activation_map_settings",
    "analyze_plaques": "get_analyze_plaque_settings",
}


def test_the_panel_factory_list_still_matches_gui_core():
    """If gui_core learns a new module, this list has to learn it too, or the
    guard below stops covering the module that was just added."""
    import inspect as _inspect
    import spacr.gui_core as GC
    src = _inspect.getsource(GC.setup_settings_panel)
    for module_id, fn_name in _PANEL_FACTORIES.items():
        assert f"'{module_id}'" in src, module_id
        assert fn_name in src, fn_name


def test_no_panel_factory_ships_a_key_that_has_no_declared_type():
    """The invariant, stated once over every module the panel can open."""
    undeclared = {}
    for module_id, fn_name in _PANEL_FACTORIES.items():
        defaults = getattr(S, fn_name)({})
        missing = sorted(k for k in defaults if k not in S.expected_types)
        if missing:
            undeclared[module_id] = missing
    assert undeclared == {}, (
        "these settings become fields but have no declared type, so "
        "check_settings drops them and gui_core's `if len(errors) > 0: "
        f"return` refuses to start the run: {undeclared}")


@pytest.mark.parametrize("fn_name", [
    "set_default_generate_barecode_mapping",
    "set_generate_training_dataset_defaults",
    "get_default_generate_activation_map_settings",
    "set_analyze_endodyogeny_defaults",
    "set_analyze_class_proportion_defaults",
    "get_check_cellpose_models_default_settings",
    "get_default_apply_cellpose_model_settings",
    "set_default_settings_preprocess_img_data",
    "set_default_settings_preprocess_generate_masks",
    "get_measure_crop_settings",
    "deep_spacr_defaults",
    "set_default_analyze_screen",
])
def test_a_module_can_be_started_with_its_own_untouched_defaults(fn_name):
    """The panel round trip, end to end, for one module.

    Build the widget map the settings panel would build from this factory,
    push it back through the validator gui_core runs, and require that every
    key comes back. Any error at all here is a module whose Run button does
    nothing until the user edits a field they were never told about.
    """
    defaults = getattr(S, fn_name)({})
    vd = _vd(**{k: ("None" if v is None else str(v))
                for k, v in defaults.items()})
    settings, errors = S.check_settings(vd, S.expected_types)
    assert errors == [], errors
    assert set(settings) == set(defaults)


@pytest.mark.parametrize("raw,want", [
    ("True", True),      # keep single-nucleus cells only
    ("False", False),    # documented trap: read as 0, removes everything
    ("10", 10),          # keep cells with 10 or fewer nuclei
    ("1000", 1000),
    ("None", None),      # filter disabled
])
def test_the_object_count_caps_keep_their_three_documented_shapes(raw, want):
    """nuclei_limit / pathogen_limit are "(int, bool, or None)" per their own
    tooltip, and four factories ship True -- but they were declared plain int,
    so check_settings rejected those modules' own defaults.

    They must NOT be declared (bool, int, NoneType) either: that triple has no
    branch of its own and falls into the generic loop, where bool() is tried
    first and bool('10') is True -- silently turning "10 or fewer nuclei" into
    "single-nucleus cells only".
    """
    for key in ("nuclei_limit", "pathogen_limit"):
        settings, errors = S.check_settings(_vd(**{key: raw}),
                                            S.expected_types)
        assert errors == [], errors
        got = settings[key]
        assert got == want and isinstance(got, type(want)), (key, got)


def test_the_object_count_caps_are_not_declared_as_the_broken_triple():
    for key in ("nuclei_limit", "pathogen_limit"):
        assert S.expected_types[key] == (bool, int)
