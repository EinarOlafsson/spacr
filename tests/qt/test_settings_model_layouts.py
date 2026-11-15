"""Table-driven tests of the settings panel's pure resolvers in ``settings_model``.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture: which defaults a module gets (built-in, external-mask
or plugin), which category layout it gets (plugin, relocations, curated
specs, hidden keys, the memo), the classifier's curated headings, the API
documentation URL, the type hint shown beside a setting, which settings get
a list widget, the count files and well keys the regression scan reads, and
the morphology an organelle slot resolves to.  No widget is built here.
"""
from __future__ import annotations

import builtins
import contextlib

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr import plugins  # noqa: E402
from spacr.qt.screens import settings_model as SM  # noqa: E402

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# shared builders
# ---------------------------------------------------------------------------

class _FakeApp:
    """The subset of ``plugins.AppContribution`` this module reads."""

    def __init__(self, defaults="ref", categories=None, docs_url=""):
        self.defaults = defaults
        self.categories = categories or {}
        self.tooltips, self.labels = {}, {}
        self.docs_url = docs_url


def _broken_registry(_key):
    raise RuntimeError("plugin registry is on fire")


@contextlib.contextmanager
def _without_spacr_settings(mp):
    """Make ``import spacr.settings`` fail for the duration of the block."""
    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "spacr.settings":
            raise ImportError("settings is half built")
        return real_import(name, *args, **kwargs)

    mp.setattr(builtins, "__import__", refuse)
    yield


# ---------------------------------------------------------------------------
# resolve_default_settings
# ---------------------------------------------------------------------------

def test_resolve_default_settings_uses_the_plugin_then_the_builtin_defaults(
        subtests, monkeypatch):
    """A plugin's callable wins and must return a dict; otherwise the built-in defaults."""
    calls = []

    def no_argument():
        calls.append("()")
        return {"src": "", "gamma": 3}

    def with_argument(settings):
        calls.append(settings)
        return {"src": "", "alpha": 1}

    cases = [
        # (id, plugin app or registry, loaded object, expected or (exc, msg), calls)
        ("defaults({}) is called first", _FakeApp(), with_argument,
         {"src": "", "alpha": 1}, [{}]),
        ("defaults() is the fallback spelling", _FakeApp(), no_argument,
         {"src": "", "gamma": 3}, ["()"]),
        ("non-callable defaults are refused", _FakeApp("pkg:not_a_function"),
         {"already": "a dict"},
         (TypeError, "Plugin defaults 'pkg:not_a_function' are not callable"), []),
        ("defaults that return a list are refused", _FakeApp("pkg:wrong"),
         lambda *_a: ["a", "b"],
         (TypeError, "Plugin defaults 'pkg:wrong' returned list, expected dict"),
         []),
    ]
    for case_id, app, loaded, expected, expected_calls in cases:
        with subtests.test(msg=case_id):
            calls.clear()
            with monkeypatch.context() as mp:
                mp.setattr(plugins, "get_app", lambda _key: app)
                mp.setattr(plugins, "load_object", lambda _ref: loaded)
                if isinstance(expected, tuple):
                    with pytest.raises(expected[0]) as caught:
                        SM.resolve_default_settings("anything")
                    assert str(caught.value) == expected[1]
                else:
                    got = SM.resolve_default_settings("anything")
                    assert got == expected
                    got["mutated"] = True
                    assert SM.resolve_default_settings("anything") == expected
            assert calls[:1] == expected_calls

    with subtests.test(msg="a registry that raises gives the built-in defaults"):
        with monkeypatch.context() as mp:
            mp.setattr(plugins, "get_app", lambda _key: None)
            builtin = SM.resolve_default_settings("measure")
            mp.setattr(plugins, "get_app", _broken_registry)
            assert SM.resolve_default_settings("measure") == builtin
        assert "src" in builtin

    with subtests.test(msg="external_masks brings its own module's defaults"):
        from spacr.external_masks import default_settings

        resolved = SM.resolve_default_settings("external_masks")
        assert resolved == default_settings({})
        assert "inputs" in resolved

    with subtests.test(msg="both classify layouts start with no source folder"):
        from spacr.settings import deep_spacr_defaults, set_default_classify

        with monkeypatch.context() as mp:
            mp.setattr(plugins, "get_app", lambda _key: None)
            for key, make in (("classify_merged", set_default_classify),
                              ("classify", deep_spacr_defaults)):
                expected = dict(make(settings={}), src=[])
                assert SM.resolve_default_settings(key) == expected


# ---------------------------------------------------------------------------
# categories_for_app
# ---------------------------------------------------------------------------

def test_categories_for_app_applies_plugin_layouts_relocations_and_hiding(
        subtests, monkeypatch):
    """Each module gets an exact, freshly copied category layout."""
    external_inputs = ["inputs", "dst", "recursive", "layout", "z_handling",
                       "plate_naming", "overwrite", "preview_only"]
    cases = [
        # (id, registry answer, app, categories, hidden, expected)
        ("a registry that raises leaves the layout as given", _broken_registry,
         "_no_builtin_uses_this", {"Main": ["src"]}, None, {"Main": ["src"]}),
        ("a plugin's categories replace the layout",
         lambda _k: _FakeApp(categories={"Tab": ("alpha", "beta")}), "x",
         {"Main": ["src"]}, None, {"Tab": ["alpha", "beta"]}),
        ("a plugin without categories keeps the layout",
         lambda _k: _FakeApp(), "x", {"Main": ["src"]}, None, {"Main": ["src"]}),
        ("external_masks moves its inputs, however often listed, to the front",
         lambda _k: None, "external_masks",
         {"Advanced": ["dst", "recursive", "dst", "keep"]}, None,
         {"Input mapping": external_inputs, "Advanced": ["keep"]}),
        ("a category emptied by hiding disappears", lambda _k: None, "_probe_app",
         {"Only the hidden one": ["hidden_one"],
          "One of each": ["hidden_one", "src"]}, {"hidden_one"},
         {"One of each": ["src"]}),
    ]
    for case_id, registry, app, categories, hidden, expected in cases:
        with subtests.test(msg=case_id):
            with monkeypatch.context() as mp:
                mp.setattr(plugins, "get_app", registry)
                if hidden is not None:
                    mp.setitem(SM._APP_HIDDEN_KEYS, app, hidden)
                got = SM.categories_for_app(app, categories)
                assert got == expected
                assert list(got) == list(expected)
                for keys in got.values():
                    keys.append("not persisted")
                assert SM.categories_for_app(app, categories) == expected

    for app, heading in (("power", "Library Design"),
                         ("classify", "Labels & Classes")):
        with subtests.test(msg=f"{app}: only leftovers get an Additional bucket"):
            placed = SM.categories_for_app(app, {})
            leftover = SM.categories_for_app(app, {"X": ["_probe_unplaced"]})
            assert heading in placed
            assert "Additional Settings" not in placed
            assert leftover["Additional Settings"] == ["_probe_unplaced"]

    with subtests.test(msg="a full memo is replaced without losing the new layout"):
        old = {(str(index),): {"Old": ["unused"]} for index in range(257)}
        with monkeypatch.context() as mp:
            mp.setattr(SM, "_CATEGORIES_FOR_APP_MEMO", old)
            mp.setattr(plugins, "get_app", lambda _key: None)
            source = {"General": ["src"]}
            first = SM.categories_for_app("probe", source)
            first["General"].append("not persisted")
            assert SM.categories_for_app("probe", source) == {"General": ["src"]}
            assert len(old) == 1


#: The settings Classify's "Evaluation" heading holds (decision 2026-09-25).
EVALUATION_KEYS = {
    "cross_validation_enabled", "cross_validation_folds", "cv_group_by",
    "holdout_plate", "nested_cv_inner_folds", "score_threshold",
    "classifier_evaluation", "evaluation_calibration", "evaluation_bins",
    "evaluation_fail_on_leakage", "leakage_audit_train_test",
    "leakage_hash_content", "leakage_require_identity",
    "cv_group_column", "well_metadata_path", "cv_exclude_batch_column",
    "prediction_treatment_column",
}


def test_the_classifier_layouts_split_evaluation_and_curate_every_setting(
        subtests):
    """Both classify layouts: two result headings, run knobs curated, no key twice."""
    # Only the alpha held-out heading may sit between the two (classify keeps
    # it beside Evaluation; the merged panel files it under Machine Learning).
    between = {"classify": ["Held-out Evaluation \u03b1"], "classify_merged": []}
    for app_key in ("classify", "classify_merged"):
        with subtests.test(msg=app_key):
            ordered = SM.categories_for_app(app_key, SM.get_categories())
            names = list(ordered)
            assert "Evaluation & Results" not in ordered
            assert names[names.index("Evaluation") + 1:
                         names.index("Results & figures")] == between[app_key]
            assert set(ordered["Evaluation"]) == EVALUATION_KEYS
            results = set(ordered["Results & figures"])
            assert {"n_top_examples", "plot", "tensorboard",
                    "intermedeate_save"} <= results
            if app_key == "classify_merged":
                assert {"cmap", "heatmap_feature", "grouping", "min_max"} <= results
            knobs = {"random_seed", "n_jobs", "pin_memory", "verbose",
                     "strict_errors", "max_failure_rate"}
            assert knobs <= set(ordered["Runtime & Reliability"])
            assert not knobs & set(ordered.get("Additional Settings", ()))
            seen = [key for keys in ordered.values() for key in keys]
            assert len(seen) == len(set(seen))


# ---------------------------------------------------------------------------
# api_docs_url
# ---------------------------------------------------------------------------

def test_api_docs_url_prefers_a_plugin_url_and_survives_the_registry(
        subtests, monkeypatch):
    """A plugin's docs URL wins; a failing registry behaves like no plugin."""
    base = "https://einarolafsson.github.io/spacr/api/"
    cases = [
        ("a plugin docs url wins",
         lambda _k: _FakeApp(docs_url="https://example.invalid/x"), "x",
         "https://example.invalid/x"),
        ("a plugin without one falls back to the index", lambda _k: _FakeApp(),
         "x", base + "index.html"),
        ("a registry that raises gives the module page", _broken_registry,
         "measure", base + "spacr/measure/index.html"),
        ("no plugin gives the module page", lambda _k: None, "measure",
         base + "spacr/measure/index.html"),
        ("an unknown module gives the API index", lambda _k: None, "x",
         base + "index.html"),
    ]
    for case_id, registry, app, expected in cases:
        with subtests.test(msg=case_id):
            with monkeypatch.context() as mp:
                mp.setattr(plugins, "get_app", registry)
                assert SM.api_docs_url(app) == expected


# ---------------------------------------------------------------------------
# _type_hint
# ---------------------------------------------------------------------------

def test_type_hint_names_the_declared_type_or_nothing(subtests, monkeypatch):
    """Declared types become words; unions are joined; ``None`` adds "(optional)"."""
    cases = [
        ("no key", "", ""),
        ("an undeclared key", "nonexistent_key_xyz", ""),
        ("an integer", "diameter", "integer"),
        ("a float", "test_split", "float"),
        ("a boolean", "verbose", "boolean"),
        ("a list", "channels", "list"),
        ("a union", "src", "string or list"),
        ("an optional string", "custom_regex", "string (optional)"),
    ]
    for case_id, key, expected in cases:
        with subtests.test(msg=case_id):
            assert SM._type_hint(key) == expected

    with subtests.test(msg="the translated hint keeps the optional marker"):
        assert SM._translated_type_hint("custom_regex", "en") == "string (optional)"
        assert SM._translated_type_hint("custom_regex", "de") == "Zeichenfolge (optional)"

    with subtests.test(msg="a settings module that will not import gives no hint"):
        with monkeypatch.context() as mp, _without_spacr_settings(mp):
            assert SM._type_hint("src") == ""


# ---------------------------------------------------------------------------
# list_shape_for
# ---------------------------------------------------------------------------

def test_list_shape_for_decides_which_settings_get_a_list_widget(
        subtests, monkeypatch):
    """(nested_capable, allow_none, element_type, container) or ``None``."""
    cases = [
        ("a scalar default keeps its ordinary widget", "_probe", "s", None),
        ("a list of ints", "channels", [1, 2], (False, False, int, list)),
        ("a list of mixed numbers is float", "_probe_x", [1.0, 2], (False, False, float, list)),
        ("a list of flags is not a list of numbers", "_probe_flags",
         [True, False], (False, False, None, list)),
        ("an undeclared tuple stays a tuple", "_probe_undeclared_tuple", (2, 3),
         (False, False, int, tuple)),
        ("nested lists of strings are nested-capable", "_p", [["a"], ["b"]],
         (True, False, str, list)),
        ("an optional declared key allows None", "custom_regex", [1, 2],
         (False, True, int, list)),
    ]
    for case_id, key, default, expected in cases:
        with subtests.test(msg=case_id):
            assert SM.list_shape_for(key, default) == expected

    with subtests.test(msg="a settings module that will not import is no declaration"):
        with monkeypatch.context() as mp, _without_spacr_settings(mp):
            assert SM.list_shape_for("channels", [1, 2]) == (False, False, int, list)
            assert SM.list_shape_for("channels", None) is None


# ---------------------------------------------------------------------------
# _count_files_of / _well_keys
# ---------------------------------------------------------------------------

def test_count_files_of_reads_the_paired_shape_then_the_legacy_one(subtests):
    """Stripped, non-blank count paths, from ``paired_data`` before ``count_data``."""
    cases = [
        ("a list of pairs", {"paired_data": [{"count": "a.csv"}]}, ["a.csv"]),
        ("one pair on its own is not a list", {"paired_data": {"count": "a.csv"}}, []),
        ("incomplete pairs are stepped over",
         {"paired_data": [{"count": None}, {"score": "s.csv"}, {"count": "  "},
                          {"count": " b.csv "}, "not a pair"]}, ["b.csv"]),
        ("blanks in the legacy list are stepped over",
         {"count_data": [None, "", "   ", "c.csv"]}, ["c.csv"]),
        ("a single legacy path is a list of one",
         {"count_data": " /screen/counts.csv "}, ["/screen/counts.csv"]),
        ("the paired shape wins over the legacy one",
         {"paired_data": [{"count": "/new.csv"}], "count_data": "/old.csv"},
         ["/new.csv"]),
        ("an empty paired shape falls back to the legacy one",
         {"paired_data": [{"count": ""}], "count_data": ["/old.csv"]}, ["/old.csv"]),
        ("no settings at all", None, []),
    ]
    for case_id, settings, expected in cases:
        with subtests.test(msg=case_id):
            assert SM._count_files_of(settings) == expected


def test_well_keys_compose_plate_row_and_column_or_refuse(subtests):
    """One key per well from ``prc``, ``plate_row`` + column, or plate/row/column."""
    sep = SM.KEY_SEPARATOR
    attributed = pd.DataFrame({"rowID": ["r1"], "columnID": ["c2"]})
    attributed.attrs[SM._FILE_PLATE] = "p9"
    cases = [
        ("a composed prc is used as it is",
         pd.DataFrame({"prc": ["p1_r1_c1", "p1_r1_c2"]}), ["p1_r1_c1", "p1_r1_c2"]),
        ("plate_row gets only the column appended",
         pd.DataFrame({"plate_row": ["p1_r1"], "columnID": ["c1"]}),
         [f"p1_r1{sep}c1"]),
        ("plate, row and column are joined",
         pd.DataFrame({"plateID": ["p2"], "rowID": ["r1"], "columnID": [3]}),
         [f"p2{sep}r1{sep}3"]),
        ("a frame without a plate takes the file's plate",
         attributed, [f"p9{sep}r1{sep}c2"]),
        ("a frame without a plate or attribute is plate1",
         pd.DataFrame({"rowID": ["r1"], "columnID": ["c2"]}),
         [f"plate1{sep}r1{sep}c2"]),
        ("none of the key columns is refused", pd.DataFrame({"count": [1]}), None),
    ]
    for case_id, frame, expected in cases:
        with subtests.test(msg=case_id):
            keys = SM._well_keys(frame)
            assert (None if keys is None else list(keys)) == expected


# ---------------------------------------------------------------------------
# organelle_morphology_now
# ---------------------------------------------------------------------------

def test_organelle_morphology_now_prefers_the_slot_then_its_type(subtests):
    """The slot's own morphology wins; else the type decides; unknowns give ``None``."""
    cases = [
        ("the slot's morphology is authoritative",
         {"organelle_morphology": "network", "organelle_type": "vesicular"}, "network"),
        ("an unknown type falls back to the slot",
         {"organelle_type": "cytoskeleton", "organelle_morphology": "ring"}, "ring"),
        ("a non-numeric diameter leaves the type to decide",
         {"organelle_type": "vesicular", "organelle_diameter": "large"}, "spots"),
        ("an unknown type and morphology narrow nothing",
         {"organelle_type": "cytoskeleton", "organelle_morphology": "blobby"}, None),
        ("nothing set narrows nothing", {}, None),
    ]
    for case_id, settings, expected in cases:
        with subtests.test(msg=case_id):
            assert SM.organelle_morphology_now("organelle", settings) == expected
