"""Regression -- the step that produces the hits -- must start from every entry point.

``spacr.ml.perform_regression`` is the last step of the pooled-screen pipeline.
All three dispatchers build its settings dict from one function:

  * Tk  -- ``gui_core.setup_settings_panel``            (settings_type 'regression')
  * Qt  -- ``qt.screens.settings_model.resolve_default_settings('regression')``
  * CLI -- ``cli.module_defaults(MODULES['regression'])``

and every one of them called ``get_perform_regression_default_settings``, which
returned 37 keys while ``perform_regression`` indexed six it did not supply --
``verbose``, ``tolerance``, ``score_column``, ``invert_dependent_variable``,
``control_wells`` and ``y_lims``. The run therefore died on
``KeyError: 'verbose'`` (ml.py:1409) *after* both input CSVs had been read and
``settings/regression.csv`` had been written, so it looked like a run that had
started cleanly and then broke on the data.

The first test here is the general form of that bug: it derives, from the AST
of ``perform_regression`` and of every helper it hands its settings dict to,
the set of keys the function indexes, and requires the defaults builder to
supply all of them. The last one runs the whole thing for real, on real CSVs,
through the CLI's own settings-resolution path.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import os
import textwrap

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


APP_KEY = "regression"

#: Where a helper named inside ``perform_regression`` may be defined. The
#: function imports them lazily from these modules in its own body.
_HELPER_MODULES = (
    "spacr.ml", "spacr.utils", "spacr.sequencing", "spacr.batch_correction",
    "spacr.settings", "spacr.toxo", "spacr.plot",
)

#: Keys ``perform_regression`` reads without a default, because it derives
#: them itself before reading them. A derived key is NOT an exemption from
#: the contract below -- it is a different way of satisfying it, and the
#: derivation is asserted rather than assumed by
#: :func:`test_the_only_keys_without_a_default_are_derived_and_written_first`.
#:
#: * ``src`` -- ``_perform_regression_set_paths`` sets it from ``count_data``
#:   (ml.py) before ``utils.save_settings`` reads it to place
#:   ``settings/regression.csv``.
#: * ``score_data`` / ``count_data`` -- the paired-input migration. One row of
#:   ``paired_data`` states one score/count relationship, and
#:   ``ml.normalize_regression_input_pairs`` unpacks it into these two lists
#:   at the top of ``perform_regression``. They are deliberately NOT
#:   defaulted: ``get_perform_regression_default_settings`` says so in as many
#:   words, because defaulting them would write the legacy pair back into
#:   every new settings CSV and undo the migration. A settings file that still
#:   carries them is migrated instead.
_DERIVED_KEYS = frozenset({"src", "score_data", "count_data"})

#: The six that were missing, and what each is for. Kept explicit so a future
#: edit that drops one is named in the failure rather than counted.
MISSING_BEFORE = {
    "verbose": "ml.py:1409 -- `if settings['verbose']:`",
    "tolerance": "ml.py:1412 -- minimum_cell_simulation(settings, tolerance=...)",
    "score_column": "ml.py:408 -- the column minimum_cell_simulation resamples",
    "invert_dependent_variable": "ml.py:1424 -- passed to process_scores",
    "control_wells": "sequencing.py:988 -- iterated by graph_sequencing_stats",
    "y_lims": "ml.py:1669 -- passed to toxo.custom_volcano_plot",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _defaults():
    from spacr.settings import get_perform_regression_default_settings
    return get_perform_regression_default_settings({})


def _tree(fn):
    return ast.parse(textwrap.dedent(inspect.getsource(fn)))


def _settings_subscripts(fn, ctx):
    """Every ``settings['literal']`` in ``fn`` used in the given context."""
    keys = set()
    for node in ast.walk(_tree(fn)):
        if (isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                # save_settings works on `settings_2`, its own copy.
                and node.value.id.startswith("settings")
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
                and isinstance(node.ctx, ctx)):
            keys.add(node.slice.value)
    return keys


#: Builtins that take the settings dict and are NOT helpers.
#:
#: The check below demands every callee be importable from
#: :data:`_HELPER_MODULES`, and its reason is in its own failure message:
#: "otherwise the keys it indexes go unchecked". A builtin indexes no keys.
#: `dict(settings)` is a copy -- `perform_regression` returns one so a caller
#: offering to re-fit has the run's own settings -- and demanding a module for
#: it asks the author to add `builtins` to the list of spaCR helper modules,
#: which is not a thing anybody should be asked to do to make a copy.
#:
#: Restricted to the ones that genuinely cannot read a key. `getattr` is
#: deliberately absent: `getattr(settings, ...)` is not how spaCR reads a
#: setting, but it is close enough to indexing that letting it through
#: silently would be a hole in exactly the direction this test guards.
_NOT_A_HELPER = frozenset({"dict", "len", "list", "sorted", "set", "tuple",
                           "bool", "repr", "print"})


def _helpers_given_the_settings_dict():
    """Names ``perform_regression`` passes its whole settings dict to.

    Nested defs are excluded: ``inspect.getsource`` already contains them, so
    their subscripts are collected with the outer function's.
    """
    from spacr.ml import perform_regression

    tree = _tree(perform_regression)
    nested = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        passed = list(node.args) + [kw.value for kw in node.keywords]
        if not any(isinstance(a, ast.Name) and a.id == "settings" for a in passed):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name and name not in nested and name not in _NOT_A_HELPER:
            names.add(name)
    return names


def _resolve(name):
    for module in _HELPER_MODULES:
        obj = getattr(importlib.import_module(module), name, None)
        if callable(obj):
            return obj
    return None


def _keys_read_by_the_whole_call(ctx=ast.Load):
    from spacr.ml import perform_regression

    functions = [perform_regression]
    for name in sorted(_helpers_given_the_settings_dict()):
        resolved = _resolve(name)
        assert resolved is not None, (
            f"perform_regression hands its settings dict to {name!r}, which is "
            f"not importable from any of {_HELPER_MODULES}. Add its module to "
            "_HELPER_MODULES -- otherwise the keys it indexes go unchecked, "
            "which is exactly how the six missing defaults survived."
        )
        functions.append(resolved)
    keys = set()
    for fn in functions:
        keys |= _settings_subscripts(fn, ctx)
    return keys


# ---------------------------------------------------------------------------
# 1. the defaults builder is a complete contract
# ---------------------------------------------------------------------------

def test_every_settings_key_perform_regression_indexes_has_a_default():
    """The general form of the bug, for every key, forever.

    ``perform_regression`` indexes its settings dict rather than ``.get()``ing
    it, so a key the builder does not supply is a KeyError on a real run --
    raised after the inputs are read and the settings snapshot is written.
    """
    read = _keys_read_by_the_whole_call()
    missing = sorted(read - set(_defaults()) - _DERIVED_KEYS)
    assert not missing, (
        "get_perform_regression_default_settings does not supply "
        f"{missing}, which perform_regression (or a helper it hands the dict "
        "to) reads with settings[...]. Every dispatcher builds the dict from "
        "that function, so regression cannot be started from Tk, Qt or the CLI."
    )
    # And the derivation really is a derivation: something must write it.
    assert read & _DERIVED_KEYS <= _keys_read_by_the_whole_call(ast.Store)


def _assert_derived_before_read(fn, deriver, keys):
    """The call that derives ``keys`` precedes every read of them in ``fn``.

    "Derived" only answers the missing-default contract if the derivation
    actually RUNS BEFORE the read. A derivation placed after would satisfy a
    membership check and still raise KeyError on a real run, which is the
    exact shape of the bug this file exists for.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    derived = [node.lineno for node in ast.walk(tree)
               if isinstance(node, ast.Call)
               and getattr(node.func, "id",
                           getattr(node.func, "attr", None)) == deriver]
    assert derived, f"{fn.__name__} no longer calls {deriver}"
    reads = [node.lineno for node in ast.walk(tree)
             if isinstance(node, ast.Subscript)
             and isinstance(node.ctx, ast.Load)
             and getattr(node.value, "id", None) == "settings"
             and isinstance(node.slice, ast.Constant)
             and node.slice.value in keys]
    assert not reads or min(derived) < min(reads), (
        f"{fn.__name__} reads settings{sorted(keys)} at line {min(reads)} "
        f"before {deriver} derives it at line {min(derived)}")


def test_the_only_keys_without_a_default_are_derived_and_written_first():
    """Derived, not forgotten -- and the difference is checked, not assumed.

    ``src`` is derived from ``count_data``; ``score_data`` and ``count_data``
    are themselves derived from ``paired_data``. "Derived" is only an answer
    to the missing-default contract if the derivation actually runs BEFORE the
    read, so the ordering is asserted rather than the mere existence of an
    assignment somewhere in the module -- an assignment placed after the read
    would satisfy a Store-membership check and still raise ``KeyError`` on a
    real run, which is the exact shape of the bug this file exists for.
    """
    from spacr.ml import perform_regression

    from spacr.ml import _perform_regression_set_paths

    read = _keys_read_by_the_whole_call()
    assert sorted(read - set(_defaults())) == sorted(_DERIVED_KEYS)

    # `src` IS DERIVED IN THE HELPER, and this used to look only inside
    # `perform_regression`. `_perform_regression_set_paths` was a nested def
    # and became a module-level function so it could be tested directly; the
    # derivation did not move, the place to look for it did, and the
    # assertion went red without anything being wrong.
    #
    # Asserted as the ORDERING, the same as the pair below, rather than as
    # "some function somewhere assigns it" -- which is the check this file
    # exists to be stricter than.
    written = (_settings_subscripts(perform_regression, ast.Store)
               | _settings_subscripts(_perform_regression_set_paths, ast.Store))
    assert "src" in written, (
        "nothing derives settings['src'], so save_settings will raise KeyError "
        "placing settings/regression.csv")
    _assert_derived_before_read(perform_regression,
                                "_perform_regression_set_paths", {"src"})

    # score_data/count_data are written by a helper, so the ordering that
    # matters is: the normalising call, then the first subscript read.
    tree = ast.parse(textwrap.dedent(inspect.getsource(perform_regression)))
    normalise = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", None))
        == "normalize_regression_input_pairs"
    ]
    assert normalise, (
        "perform_regression no longer calls normalize_regression_input_pairs, "
        "so score_data/count_data are read but never derived. Either restore "
        "the call or give them defaults.")
    reads = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.ctx, ast.Load)
        and getattr(node.value, "id", None) == "settings"
        and isinstance(node.slice, ast.Constant)
        and node.slice.value in {"score_data", "count_data"}
    ]
    assert reads, "expected perform_regression to read the derived pair"
    assert min(normalise) < min(reads), (
        f"perform_regression reads settings['score_data'/'count_data'] at "
        f"line {min(reads)} of its own source but only derives them at line "
        f"{min(normalise)}, so a settings dict built from the defaults raises "
        f"KeyError before the derivation runs.")


@pytest.mark.parametrize("key", sorted(MISSING_BEFORE))
def test_the_six_keys_that_were_missing_are_supplied(key):
    assert key in _defaults(), (
        f"{key} is read at {MISSING_BEFORE[key]} but is not defaulted"
    )


def test_the_missing_six_have_values_their_readers_accept():
    """Present is not enough -- each has a shape its reader requires."""
    from spacr.toxo import _normalize_y_lims

    defaults = _defaults()

    # ml.py:1409 `if settings['verbose']:` -- and the verbose branch display()s
    # the whole per-object score table, so a screen must not default to it.
    assert defaults["verbose"] is False

    # minimum_cell_simulation: int == percent, float == fraction, else ValueError.
    assert isinstance(defaults["tolerance"], (int, float))
    assert 0 < defaults["tolerance"] <= 100

    # The column it resamples has to exist in the score table it is given.
    assert defaults["score_column"] == defaults["dependent_variable"]

    # process_scores accepts False/0, True/1 or -1 and raises on anything else.
    assert defaults["invert_dependent_variable"] in (False, 0, True, 1, -1)

    # graph_sequencing_stats does `for c in settings['control_wells']`, so None
    # -- what the invasion assay defaults the same key name to -- would raise.
    assert isinstance(defaults["control_wells"], list)
    iter(defaults["control_wells"])

    # toxo.custom_volcano_plot normalises this and raises on any other shape.
    assert _normalize_y_lims(defaults["y_lims"], pd.Series([1.0, 2.0]))


def test_control_wells_names_the_same_wells_as_filter_value():
    """The threshold sweep and the score filter must drop the same wells.

    ``graph_sequencing_stats`` drops ``control_wells`` from the count table
    before it picks ``fraction_threshold``; ``ml.clean_controls`` drops
    ``filter_value`` from the score table. If they disagree, the threshold is
    fitted on wells the regression never sees.
    """
    defaults = _defaults()
    assert defaults["control_wells"] == defaults["filter_value"]

    chosen = _defaults()
    chosen["filter_value"] = ["c11", "c12"]
    from spacr.settings import get_perform_regression_default_settings
    assert get_perform_regression_default_settings(
        {"filter_value": ["c11", "c12"]})["control_wells"] == ["c11", "c12"]
    # A non-list filter_value (the str form clean_controls also accepts) must
    # still leave something iterable behind.
    assert get_perform_regression_default_settings(
        {"filter_value": "c1"})["control_wells"] == []


def test_score_column_follows_a_chosen_dependent_variable():
    """Otherwise the cell-count simulation describes a different measurement."""
    from spacr.settings import get_perform_regression_default_settings

    chosen = get_perform_regression_default_settings(
        {"dependent_variable": "pathogen_nucleus_shortest_distance"})
    assert chosen["score_column"] == "pathogen_nucleus_shortest_distance"
    # An explicit score_column still wins.
    explicit = get_perform_regression_default_settings(
        {"dependent_variable": "recruitment", "score_column": "pred"})
    assert explicit["score_column"] == "pred"


def test_quantile_regression_still_clears_agg_type():
    """The one piece of logic in the builder, unchanged by the new defaults."""
    from spacr.settings import get_perform_regression_default_settings

    assert get_perform_regression_default_settings(
        {"regression_type": "quantile"})["agg_type"] is None


# ---------------------------------------------------------------------------
# 2. all three dispatchers really do resolve the same dict
# ---------------------------------------------------------------------------

def test_the_cli_module_resolves_the_defaults_builder():
    from spacr.cli import MODULES, module_defaults

    module = MODULES[APP_KEY]
    assert module.defaults == "get_perform_regression_default_settings"
    assert set(module_defaults(module)) == set(_defaults())


def test_the_qt_panel_resolves_the_same_defaults():
    pytest.importorskip("PySide6")
    from spacr.qt.screens.settings_model import resolve_default_settings

    assert set(resolve_default_settings(APP_KEY)) == set(_defaults())


def test_the_tk_panel_resolves_the_same_defaults():
    """Read from source: importing gui_core needs a Tk display."""
    import spacr.gui_core

    source = inspect.getsource(spacr.gui_core.setup_settings_panel)
    assert ("settings_type == 'regression'" in source
            or 'settings_type == "regression"' in source)
    assert "get_perform_regression_default_settings(settings={})" in source


def test_the_resolved_dict_passes_its_own_pre_flight():
    """Stock defaults must not be reported as a problem by the CLI's own check.

    Two of the six new keys are typed for the first time, and one of them --
    ``control_wells`` -- is shared with the invasion assay, where it is
    ``(list, None)``. A type entry that disagreed with the default would make
    every regression run fail pre-flight.
    """
    from spacr.cli import MODULES, module_defaults
    from spacr.validate import validate_settings

    settings = module_defaults(MODULES[APP_KEY])
    typed = {p.setting for p in validate_settings(settings, APP_KEY)
             if p.is_error and p.setting in settings}
    assert typed <= {"score_data", "count_data"}, (
        f"stock regression defaults fail their own pre-flight on {sorted(typed)}"
    )


# ---------------------------------------------------------------------------
# 2b. the Tk panel coerces the two newly-typed shapes correctly
# ---------------------------------------------------------------------------
#
# ``check_settings`` walks ``expected_types`` and parses each widget's raw
# string. Declaring a type therefore has teeth: the generic
# "try each type in the tuple" fallback at the bottom of that function reaches
# ``bool('False')`` -- True -- for ``(bool, int)``, and ``list('[0, 5]')`` --
# ['[', '0', ',', ' ', '5', ']'] -- for ``(list, None)``. Both now have their
# own branch. The second one also repairs ``x_lim``, ``control_wells`` and
# ``filter_min_max``, which carried the same declared type all along.

class _Var:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


def _widget_map(**values):
    return {k: (None, None, _Var(v), None) for k, v in values.items()}


@pytest.mark.parametrize(("text", "expected"), [
    ("False", False), ("false", False), ("no", False), ("0", 0),
    ("True", True), ("true", True), ("yes", True), ("1", 1),
    ("-1", -1),
])
def test_invert_dependent_variable_is_not_read_as_bool_of_the_string(text, expected):
    from spacr.settings import check_settings, expected_types

    settings, errors = check_settings(
        _widget_map(invert_dependent_variable=text), expected_types)
    assert not errors
    assert settings["invert_dependent_variable"] == expected
    # False and 0 must not come back True, which is what bool('False') gives.
    assert bool(settings["invert_dependent_variable"]) is bool(expected)


def test_invert_dependent_variable_rejects_a_word_it_cannot_read():
    from spacr.settings import check_settings, expected_types

    settings, errors = check_settings(
        _widget_map(invert_dependent_variable="maybe"), expected_types)
    assert errors and "invert_dependent_variable" in errors[0]


@pytest.mark.parametrize("text", ["", "None"])
def test_an_emptied_invert_field_is_none_not_true(text):
    """'' and 'None' become None at the top of check_settings."""
    from spacr.settings import check_settings, expected_types

    settings, errors = check_settings(
        _widget_map(invert_dependent_variable=text), expected_types)
    assert not errors
    assert settings["invert_dependent_variable"] is None


def test_the_new_branches_do_not_swallow_their_neighbours():
    """The two new elifs sit in a chain; the ones after them still run."""
    from spacr.settings import check_settings, expected_types

    assert expected_types["transform"] == (str, type(None))
    settings, errors = check_settings(_widget_map(transform="log"),
                                      expected_types)
    assert not errors
    assert settings["transform"] == "log"


@pytest.mark.parametrize(("key", "text", "expected"), [
    ("y_lims", "[0, 5]", [0, 5]),
    ("y_lims", "[[0, 5], [40, 60]]", [[0, 5], [40, 60]]),
    ("y_lims", "(0, 5)", [0, 5]),
    ("x_lim", "[-0.5, 0.5]", [-0.5, 0.5]),
    ("control_wells", "['c1', 'c2']", ["c1", "c2"]),
])
def test_list_or_none_settings_parse_instead_of_being_split_into_characters(
        key, text, expected):
    from spacr.settings import check_settings, expected_types

    settings, errors = check_settings(_widget_map(**{key: text}),
                                      expected_types)
    assert not errors
    assert settings[key] == expected


@pytest.mark.parametrize("text", ["None", ""])
def test_list_or_none_settings_accept_none(text):
    from spacr.settings import check_settings, expected_types

    settings, errors = check_settings(_widget_map(y_lims=text), expected_types)
    assert not errors
    assert settings["y_lims"] is None


@pytest.mark.parametrize("text", ["not a list", "5"])
def test_list_or_none_settings_reject_anything_else(text):
    from spacr.settings import check_settings, expected_types

    settings, errors = check_settings(_widget_map(y_lims=text), expected_types)
    assert errors and "y_lims" in errors[0]


# ---------------------------------------------------------------------------
# 3. the real thing, end to end, through the CLI's settings path
# ---------------------------------------------------------------------------

GENES = ("000000", "233460", "239740", "111111")
ROWS = ("r1", "r2", "r3")
COLS = ("c1", "c2", "c3", "c4", "c5", "c6")


def _write_screen(root):
    """A tiny but real pooled screen: per-object scores + per-well gRNA counts."""
    sdir = root / "scores"
    cdir = root / "counts"
    sdir.mkdir()
    cdir.mkdir()

    rng = np.random.default_rng(0)
    scores = []
    for row in ROWS:
        for col in COLS:
            base = float(rng.uniform(0.2, 0.8))
            for _ in range(6):
                scores.append({
                    "plateID": "plate1", "rowID": row, "columnID": col,
                    "fieldID": "f1",
                    "pred": float(np.clip(base + rng.normal(0, 0.1), 0.02, 0.98)),
                })
    score_csv = sdir / "xgb_scores.csv"
    pd.DataFrame(scores).to_csv(score_csv, index=False)

    rng = np.random.default_rng(1)
    counts = []
    for row in ROWS:
        for col in COLS:
            for gene in GENES:
                for i in range(1, 4):
                    counts.append({
                        "plateID": "plate1", "rowID": row, "columnID": col,
                        "grna": f"TGGT1_{gene}_{i}",
                        "count": int(rng.integers(20, 400)),
                    })
    count_csv = cdir / "counts.csv"
    pd.DataFrame(counts).to_csv(count_csv, index=False)
    return str(score_csv), str(count_csv), cdir


@pytest.fixture(autouse=True)
def _no_figure_leak():
    yield
    plt.close("all")


def test_regression_runs_end_to_end_from_the_cli_settings_path(tmp_path):
    """The caller, not the callee: ``spacr-run regression``'s own resolution.

    Nothing is stubbed. ``minimum_cell_simulation`` (which reads ``tolerance``
    and ``score_column``) and ``graph_sequencing_stats`` (which iterates
    ``control_wells``) both run for real.

    ``min_cell_count`` USED TO BE None by default, which is what made the
    simulation run here without being asked for. It is 100 now -- a deliberate
    change, requested in d6eb6ca3 along with transform=log and
    multiple_testing_method=none, on the ground that below 100 cells a well's
    score is noise dressed as a measurement. So this test now does two things
    rather than one: it pins the default a user who edits nothing gets, and it
    then asks for None explicitly, because the simulation path is the one that
    reads ``tolerance`` and ``score_column`` and it would otherwise stop being
    covered by anything.

    Before the fix this raised ``KeyError: 'verbose'`` with
    ``settings/regression.csv`` already on disk and not one result file
    written.
    """
    from spacr.cli import MODULES, resolve_settings
    from spacr.ml import perform_regression

    score_csv, count_csv, cdir = _write_screen(tmp_path)
    settings_csv = tmp_path / "regression.csv"
    pd.DataFrame(
        [("score_data", repr([score_csv])),
         ("count_data", repr([count_csv])),
         # No metadata CSVs to join, so the Toxo reports have nothing to draw.
         ("toxo", "False"),
         ("metadata_files", "[]")],
        columns=["Key", "Value"],
    ).to_csv(settings_csv, index=False)

    settings = resolve_settings(MODULES[APP_KEY], str(settings_csv))
    assert settings["min_cell_count"] == 100, (
        "the requested default; if this moves, move it here deliberately")
    assert settings["fraction_threshold"] is None

    # …and now drive the simulated path, which the default no longer reaches.
    settings["min_cell_count"] = None

    np.random.seed(0)
    out = perform_regression(settings)

    # The fit comes back with the coefficients, not just the verdict. This
    # was {"results", "significant"} until c0db2f48: a consumer could say WHAT
    # was significant and nothing about whether the fit deserved to be
    # believed. `model` and `model_data` are what the QC suite reads to get
    # R-squared, residuals and the design that actually reached the fit, so
    # dropping either would silently take the diagnostics away again -- which
    # is why this stays an exact set rather than a subset check.
    assert set(out) == {"results", "significant", "model", "model_data",
                        "regression_type", "res_folder", "settings"}

    # `settings` is the run's OWN dict, so a caller offering to re-fit the
    # same screen through a different model has it without reading a file --
    # the shared settings/ copy is overwritten by every later run of the same
    # screen, so on a second run the file describes the wrong one.
    assert out["settings"]["regression_type"] == "ols"
    assert out["settings"] is not settings, (
        "the run handed back the caller's own dict, so mutating the copy "
        "would reach back into the settings the caller still holds")

    # ASKED FOR, NOT SPELLED OUT. This was
    # `results/xgb_scores/ols/list` -- the four-level path from before the
    # output rule became `<count folder>/results/<type>`, with `_1`, `_2` for
    # a repeat. Same staleness as test_regression_types.py had, and the same
    # fix: the run says where it wrote.
    res = out["res_folder"]
    assert os.path.dirname(res) == os.path.join(str(cdir), "results"), res
    results = pd.read_csv(os.path.join(res, "results.csv"))
    assert len(results) > 0
    for name in ("results_gene.csv", "results_grna.csv",
                 "results_significant.csv", "regression_data.csv"):
        assert os.path.isfile(os.path.join(res, name)), name

    # The simulation and the threshold sweep both produced a value.
    assert isinstance(settings["min_cell_count"], (int, float, np.integer))
    assert isinstance(settings["fraction_threshold"], (int, float, np.floating))

    # And the snapshot beside the results records all six.
    snapshot = pd.read_csv(os.path.join(str(cdir), "settings", "regression.csv"))
    assert set(MISSING_BEFORE) <= set(snapshot["Key"])
