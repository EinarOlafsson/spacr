"""The test suite's own hygiene, enforced by AST over ``tests/``.

Three ways a test can look green while testing nothing, each of which this
suite has actually shipped:

1. **No assertion at all.** The test calls the function and checks nothing.
   ``tests/test_all_plotting_functions.py`` carried fourteen of these; every
   one of them passed against a blank figure.
2. **A broad ``except`` that turns a failure into a ``pytest.skip``.** A
   self-skip makes "this machine cannot run the test" and "the product is
   broken" indistinguishable. ``test_image_umap_end_to_end`` reported skipped
   for its entire life while ``generate_image_umap`` was never once called.
3. **A machine-specific absolute path.** A test whose precondition is
   ``/home/<someone>/datasets`` runs on exactly one computer and reports
   green everywhere else.

Each rule carries a RATCHET list: the violations that existed when the rule
was written, so the rule passes today and fails the moment a NEW one appears.
The lists may only shrink. They are keyed by (file, function) rather than by
file, so adding an assertion-free test to an already-listed module still
fails, and an entry naming a function that no longer exists is ignored rather
than fatal (renaming or deleting a test must never break this file).

Where one key can cover more than one violation -- a function holding two
broad excepts, a module holding two identically-named mocks -- the ratchet
records a COUNT, not just the key. A set under-reports: the broad-skip list
held 38 (file, function) pairs against 46 actual handlers, so a second
failure-swallowing handler could be added to any listed function and the
rule would still be green.

Fix, do not extend. Every entry below is a test that is not currently earning
its green.
"""
from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

#: Context managers that ARE an assertion: entering them asserts that the body
#: raises/warns.
_ASSERTING_CONTEXTS = frozenset({"raises", "warns", "deprecated_call"})


def _test_modules():
    """Every python module under tests/, path-relative to tests/."""
    return sorted(TESTS_DIR.rglob("*.py"))


def _rel(path):
    return str(Path(path).relative_to(TESTS_DIR))


def _parse(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _call_name(node):
    """The bare name of whatever a Call node calls (``a.b.c()`` -> ``'c'``)."""
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")


def _functions(tree):
    return [n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _direct_assertions(fn):
    """``(has_direct_assertion, names_it_calls)`` for one function body."""
    called = set()
    found = False
    for node in ast.walk(fn):
        if node is fn:
            continue
        if isinstance(node, ast.Assert):
            found = True
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                ctx = item.context_expr
                if isinstance(ctx, ast.Call) and _call_name(ctx) in _ASSERTING_CONTEXTS:
                    found = True
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            called.add(name)
            # `assertEqual`, `assert_allclose`, `np.testing.assert_*`,
            # `pytest.fail`, `pytest.raises(...)` used bare.
            if name.startswith("assert") or name in _ASSERTING_CONTEXTS or name == "fail":
                found = True
    return found, called


def _asserts_something(fn, helpers, memo, stack=()):
    """True when ``fn`` asserts, directly or via a helper in the same module.

    Resolving one module's own helpers matters: a test whose whole body is
    ``_nonempty_file(path)`` or ``_assert_masks_match(a, b)`` is a real test,
    and a rule that could not see through the helper would push authors to
    inline everything.
    """
    direct, called = _direct_assertions(fn)
    if direct:
        return True
    for name in called:
        if name in stack:
            continue                      # recursive helper; already walked
        helper = helpers.get(name)
        if helper is None:
            continue
        if name in memo:
            if memo[name]:
                return True
            continue
        result = _asserts_something(helper, helpers, memo, stack + (name,))
        memo[name] = result
        if result:
            return True
    return False


def _is_broad_skip(handler):
    """True when this ``except`` catches everything and turns it into a skip."""
    caught = handler.type
    broad = caught is None or (isinstance(caught, ast.Name)
                               and caught.id in ("Exception", "BaseException"))
    if not broad:
        return False
    if any(isinstance(n, ast.Raise) for n in ast.walk(handler)):
        return False                      # re-raises: the failure survives
    return any(isinstance(n, ast.Call)
               and _call_name(n) in ("skip", "importorskip", "xfail")
               for n in ast.walk(handler))


def _broad_skip_handlers(fn):
    """Line numbers of broad ``except`` blocks that skip without re-raising."""
    return [node.lineno for node in ast.walk(fn)
            if isinstance(node, ast.ExceptHandler) and _is_broad_skip(node)]


# ---------------------------------------------------------------------------
# Rule 1 — every test asserts something
# ---------------------------------------------------------------------------

#: Tests that assert nothing today. Snapshot taken when this file was written;
#: this list may only shrink. Fix the test, do not add to it.
ASSERTION_FREE_RATCHET = {
    "qt/test_console_thread_safety.py": {
        "test_a_collected_console_target_is_dropped_rather_than_resurrected",
        "test_the_relay_swallows_an_exploding_console",
    },
    "qt/test_cov_qt_app.py": {
        "test_a_zoo_request_is_dropped_if_compare_cannot_be_configured",
        "test_apply_demo_to_a_screen_that_supports_nothing_is_a_no_op",
        "test_missing_font_directory_is_not_an_error",
    },
    "qt/test_db_browser.py": {
        "test_thread_startup_has_no_signal_disconnect_warning",
    },
    "qt/test_dna_rain.py": {
        "test_a_broken_theme_lookup_does_not_break_the_switch",
        "test_settings_bar_emits_its_signals",
        "test_unrelated_change_events_are_ignored",
    },
    "qt/test_e2e_pipeline.py": {
        "test_measure_crop_runs_on_measure_demo",
        "test_preprocess_generate_masks_runs_on_mask_demo",
    },
    "qt/test_home_variants.py": {
        "test_every_categorisation_covers_every_app",
        "test_prune_stale_dirs_is_a_no_op_without_a_versions_dir",
    },
    "qt/test_live_palette_consumers.py": {
        "test_qt_app_import_has_no_frozen_palette_warning",
    },
    "qt/test_live_preview.py": {
        "test_switch_modes_no_crash",
    },
    "qt/test_model_compare_screen.py": {
        "test_closing_survives_a_thread_whose_c_plus_plus_side_is_already_gone",
    },
    "qt/test_photo_themes.py": {
        "test_no_role_lookup_raises_for_any_theme",
    },
    "qt/test_preferences.py": {
        "test_apply_preferences_to_app_does_not_raise",
        "test_preferences_dialog_builds_and_closes",
    },
    "qt/test_shortcuts_and_notify.py": {
        "test_announce_pipeline_finished_does_not_raise",
        "test_show_cheat_sheet_opens_and_closes",
    },
    "qt/test_space_theme.py": {
        "test_refreshing_a_broken_window_is_swallowed",
    },
    "qt/test_track_previews.py": {
        "test_motility_propagate_failure_is_swallowed",
        "test_propagate_failure_is_swallowed",
    },
    "qt/test_tutorial_engine.py": {
        "test_cursor_overlay_draws_on_pixmap",
        "test_highlight_overlay_draws_on_pixmap",
    },
    "qt/test_widgets.py": {
        "test_tile_emits_clicked",
    },
    "test_analysis_modules_t10.py": {
        "test_analyze_entrypoint_smoke",
    },
    "test_cli.py": {
        "test_noshow_survives_a_matplotlib_that_raises",
    },
    "test_cov_deep_spacr_smoothgrad_examples.py": {
        "test_visualize_smooth_grad_handles_image_size_mismatch",
    },
    "test_cov_deep_spacr_train_test_entry.py": {
        "test_split_dir",
    },
    "test_coverage_fill_cellpose_gpu_funcs.py": {
        "test_channel_selection_per_model",
        "test_check_cellpose_models",
        "test_custom_model_present",
        "test_generate_masks_non_normalize_resize",
        "test_generate_plot_resize",
        "test_grayscale_and_verbose",
        "test_identify_verbose_resize",
        "test_no_cuda_branch",
        "test_non_normalize_resize_path",
        "test_normalize_path_runs",
    },
    "test_coverage_fill_cli_init.py": {
        "test_timer_noop_when_not_started",
    },
    "test_coverage_fill_gui_elements.py": {
        "test_modify_figure_properties_none",
        "test_save_figure_as_format_cancelled",
    },
    "test_coverage_fill_io.py": {
        "test_save_settings_to_db",
    },
    "test_coverage_fill_measure_object.py": {
        "test_validate_organelle_settings_ok",
    },
    "test_coverage_fill_plot2.py": {
        "test_plot_histograms_and_stats",
    },
    "test_coverage_fill_toxo.py": {
        "test_go_term_enrichment_by_column",
        "test_plot_gene_phenotypes",
    },
    "test_coverage_fill_utils8.py": {
        "test_model_metrics",
    },
    "test_deep_spacr_inference.py": {
        "test_generate_activation_map_gradcam",
        "test_generate_activation_map_saliency",
    },
    "test_dry_run.py": {
        "test_measure_crop_dry_run_never_starts_a_worker_pool",
        "test_measure_crop_dry_run_returns_before_any_heavy_import",
        "test_preprocess_generate_masks_dry_run_loads_no_model",
    },
    "test_e2e_real_pipeline.py": {
        "test_stage6_core_umap_and_graphs",
    },
    "test_errors.py": {
        "test_to_dict_is_json_serialisable_even_with_tracebacks",
    },
    "test_gui_elements.py": {
        "test_spacr_switch_toggles",
    },
    "test_io_helpers_more.py": {
        "test_create_movies_from_npy_per_channel",
    },
    "test_more_helpers.py": {
        "test_utils_check_index_short_prefix_ok",
    },
    "test_more_helpers_2.py": {
        "test_plot_histogram_dst_none_still_returns",
    },
    "test_more_helpers_3.py": {
        "test_utils_copy_images_to_consolidated_no_op_on_empty_map",
    },
    "test_more_helpers_5.py": {
        "test_toxo_plot_gene_heatmaps_smoke",
        "test_toxo_plot_gene_phenotypes_smoke",
    },
    "test_more_helpers_6.py": {
        "test_utils_check_index_various_element_counts",
    },
    "test_more_helpers_7.py": {
        "test_plot_lorenz_curves_smoke_synthetic_csvs",
        "test_plot_visualize_masks_binary_and_multilabel",
        "test_plot_visualize_masks_runs_on_three_masks",
    },
    "test_more_helpers_8.py": {
        "test_settings_categories_dict_no_duplicate_setting_across_groups",
    },
    "test_new_widgets.py": {
        "test_spacr_toggle_command_swallows_exceptions",
    },
    "test_object.py": {
        "test_validate_organelle_settings_accepts_valid_combos",
    },
    "test_pipeline_training_analysis.py": {
        "test_analyze_recruitment_runs_on_pipeline_db",
    },
    "test_plot.py": {
        "test_generate_plate_heatmap_output_is_plottable",
    },
    "test_python314_optional_native.py": {
        "test_core_modules_do_not_eagerly_import_optional_native_features",
    },
    "test_regressions.py": {
        "test_measure_module_imports",
    },
    "test_smoke.py": {
        "test_module_imports",
        "test_source_parses",
    },
    "test_tstack.py": {
        "test_the_guard_accepts_a_list_of_2d_frames",
    },}

#: Total entries above. Pinned so a fix that "resolves" a violation by adding
#: two more cannot pass.
ASSERTION_FREE_CEILING = 78


def _assertion_free_tests():
    """Every ``test_*`` function in the suite that asserts nothing."""
    offenders = []
    for path in _test_modules():
        tree = _parse(path)
        helpers = {f.name: f for f in _functions(tree)}
        memo = {}
        for fn in _functions(tree):
            if not fn.name.startswith("test"):
                continue
            if not _asserts_something(fn, helpers, memo):
                offenders.append((_rel(path), fn.name, fn.lineno))
    return offenders


def test_no_new_assertion_free_tests():
    """A test that asserts nothing cannot fail, so it is not a test."""
    offenders = _assertion_free_tests()
    unlisted = [(f, n, ln) for f, n, ln in offenders
                if n not in ASSERTION_FREE_RATCHET.get(f, ())]
    assert not unlisted, (
        "these tests call code and assert nothing:\n" +
        "\n".join(f"  {f}:{ln} {n}()" for f, n, ln in unlisted) +
        "\n\nGive each one a real assertion. `assert x is not None` does not "
        "count -- assert the shape, the values, the file that was written. "
        "Adding them to ASSERTION_FREE_RATCHET is not a fix."
    )


def test_the_assertion_free_ratchet_only_shrinks():
    """The snapshot is a debt ceiling, not a budget to spend."""
    total = sum(len(v) for v in ASSERTION_FREE_RATCHET.values())
    assert total <= ASSERTION_FREE_CEILING, (
        f"ASSERTION_FREE_RATCHET grew to {total} entries (ceiling "
        f"{ASSERTION_FREE_CEILING}). Entries come off this list, never on.")
    # And nothing on it may be stale-by-file: a whole module disappearing is
    # worth noticing, unlike a single renamed test.
    missing = [f for f in ASSERTION_FREE_RATCHET if not (TESTS_DIR / f).is_file()]
    assert not missing, (
        f"ASSERTION_FREE_RATCHET names modules that no longer exist: {missing}")


# ---------------------------------------------------------------------------
# Rule 2 — a broad except may not hide a failure behind a skip
# ---------------------------------------------------------------------------

#: The scope name used for a handler that is not inside any function.
MODULE_SCOPE = "<module>"

#: Scopes containing ``except Exception: ... pytest.skip(...)`` with no
#: re-raise, and HOW MANY such handlers each holds. Most are harmless import
#: guards; three of them were hiding real product bugs when this rule was
#: written. Narrow the exception type (an import guard wants
#: ``pytest.importorskip``; a download guard wants ``OSError``) rather than
#: adding entries here.
#:
#: The counts are load-bearing. As a set of (file, function) pairs this list
#: held 38 entries against 46 real handlers, so a second failure-swallowing
#: handler could be dropped into any already-listed function for free.
#:
#: ``"<module>"`` entries are handlers at module scope -- the
#: ``try: import spacr.gui_elements / except Exception: pytest.skip(...,
#: allow_module_level=True)`` shape. They are the highest-blast-radius form of
#: this pattern (one of them turns a genuine import-time product failure into
#: a whole FILE reporting skipped) and the previous implementation, which only
#: walked function bodies, could not see a single one of them.
BROAD_SKIP_RATCHET = {
    "qt/test_e2e_pipeline.py": {
        "_require": 1,
    },
    "qt/test_gui_run_and_console.py": {
        "_require_gpu_cellpose": 2,
    },
    "test_analysis_modules_t10.py": {
        "test_analyze_entrypoint_smoke": 1,
    },
    "test_analysis_submodules_real_data.py": {
        "_require_gpu_cellpose": 2,
        "test_apply_cellpose_model_writes_results": 1,
        "test_count_phenotypes_real_db": 1,
        "test_train_cellpose_writes_model": 1,
    },
    "test_cov_gui_elements_widgets_card_toggle.py": {
        MODULE_SCOPE: 1,
    },
    "test_cov_gui_elements_widgets_progress.py": {
        MODULE_SCOPE: 1,
    },
    "test_coverage_fill_io.py": {
        "test_save_settings_to_db": 1,
    },
    "test_coverage_fill_settings.py": {
        "test_defaults_function_populates_dict": 1,
    },
    "test_coverage_fill_toxo.py": {
        "test_plot_gene_phenotypes": 1,
    },
    "test_e2e_real_dataset.py": {
        "test_e2e_real_stage_2_measure": 2,
    },
    "test_extended_coverage.py": {
        "test_gui_elements_set_element_size_returns_dict": 1,
        "test_gui_main_app_carries_color_settings": 1,
        "test_gui_main_app_constructs": 1,
    },
    "test_full_pipeline_e2e.py": {
        "_require_gpu_stack": 2,
        "test_stage_2_measure_and_crop": 1,
        "test_stage_4_train_resnet_10_epochs": 1,
        "test_stage_5_apply_model_to_full_dataset": 1,
    },
    "test_gui_elements.py": {
        MODULE_SCOPE: 1,
    },
    "test_gui_utils_and_core.py": {
        MODULE_SCOPE: 1,
    },
    "test_hf_dataset.py": {
        "test_cellposesam_on_hf_toxo_mito_field": 1,
    },
    "test_hf_e2e_integration.py": {
        "test_hf_e2e_measure_stage": 2,
    },
    "test_io_classes_more.py": {
        "test_save_mask_timelapse_as_gif": 1,
    },
    "test_io_helpers_more.py": {
        "test_create_movies_from_npy_per_channel": 1,
    },
    "test_more_helpers_5.py": {
        "test_toxo_plot_gene_heatmaps_smoke": 1,
        "test_toxo_plot_gene_phenotypes_smoke": 1,
    },
    "test_more_helpers_6.py": {
        "test_object_preprocess_batch_rolling_ball_only": 1,
    },
    "test_pipeline_e2e.py": {
        "test_apply_model_runs_on_generated_pngs": 1,
        "test_generate_dataset_creates_datasets_folder": 1,
    },
    "test_pipeline_training_analysis.py": {
        "test_analyze_recruitment_runs_on_pipeline_db": 1,
        "test_ml_analysis_random_forest_variant": 1,
        "test_ml_analysis_returns_dataframe_and_importances": 1,
        "test_train_test_model_produces_a_saved_model": 1,
    },
    "test_real_data_image_modules.py": {
        "_require_gpu_cellpose": 2,
        "test_module_measure_crop_writes_measurements_db": 1,
    },
    "test_sequencing.py": {
        "test_generate_barecode_mapping_end_to_end": 1,
    },
    "test_submodules.py": {
        "test_count_phenotypes_produces_csv": 1,
    },
    "test_utils_db_activation.py": {
        "test_organelle_diagnostic_modes": 1,
    },
    "test_utils_training_advice.py": {
        "test_suggest_training_changes_missing_csvs": 1,
    },
    "test_v1_v2_parity.py": {
        "_require_gpu_cellpose": 3,
    },}

#: Total HANDLERS above, not keys. 46 in function bodies + 4 at module scope.
BROAD_SKIP_CEILING = 50


def _scan_broad_skips(tree):
    """``(scope, lineno)`` for every broad-except-to-skip handler in one tree.

    ``scope`` is the enclosing function's name, or :data:`MODULE_SCOPE` when
    the handler is not inside one. Module scope has to be walked separately:
    ``_functions(tree)`` returns function bodies only, so an import guard at
    the top of a file -- which skips the ENTIRE module on any exception -- was
    invisible to this rule for its whole life.
    """
    sites = []
    functions = _functions(tree)
    # Handlers reachable from a function body, by identity. `tree` is held by
    # the caller for the whole call, so these ids stay valid.
    inside_a_function = {id(n) for fn in functions for n in ast.walk(fn)
                         if isinstance(n, ast.ExceptHandler)}
    for fn in functions:
        for lineno in _broad_skip_handlers(fn):
            sites.append((fn.name, lineno))
    for node in ast.walk(tree):
        if (isinstance(node, ast.ExceptHandler)
                and id(node) not in inside_a_function
                and _is_broad_skip(node)):
            sites.append((MODULE_SCOPE, node.lineno))
    return sites


def _broad_skip_sites():
    """``(file, scope, lineno)`` for every broad-except-to-skip handler."""
    sites = []
    for path in _test_modules():
        tree = _parse(path)
        for scope, lineno in _scan_broad_skips(tree):
            sites.append((_rel(path), scope, lineno))
    return sites


def test_no_new_failure_swallowing_skips():
    """``except Exception: pytest.skip(...)`` reports a bug as an excuse."""
    found = {}
    for f, scope, lineno in _broad_skip_sites():
        found.setdefault((f, scope), []).append(lineno)
    over_budget = []
    for (f, scope), linenos in sorted(found.items()):
        allowed = BROAD_SKIP_RATCHET.get(f, {}).get(scope, 0)
        if len(linenos) > allowed:
            over_budget.append((f, scope, sorted(linenos), allowed))
    assert not over_budget, (
        "these scopes turn any failure into a skip more often than the "
        "ratchet allows:\n" +
        "\n".join(f"  {f}: {scope} has {len(lns)} handler(s) at lines {lns}, "
                  f"ratchet allows {allowed}"
                  for f, scope, lns, allowed in over_budget) +
        "\n\nCatch the specific exception the environment can actually raise "
        "(pytest.importorskip for a missing package, OSError for a download), "
        "or re-raise. A skip must never be reachable from a bug in spaCR. "
        f"A scope of {MODULE_SCOPE!r} means the guard is at module level and "
        "skips the whole file."
    )


def test_the_broad_skip_ratchet_only_shrinks():
    total = sum(sum(scopes.values()) for scopes in BROAD_SKIP_RATCHET.values())
    assert total <= BROAD_SKIP_CEILING, (
        f"BROAD_SKIP_RATCHET grew to {total} handlers (ceiling "
        f"{BROAD_SKIP_CEILING}).")
    missing = [f for f in BROAD_SKIP_RATCHET if not (TESTS_DIR / f).is_file()]
    assert not missing, (
        f"BROAD_SKIP_RATCHET names modules that no longer exist: {missing}")


def test_the_broad_skip_rule_can_see_a_module_level_guard():
    """The rule's own blind spot, pinned.

    ``_broad_skip_sites`` used to walk function bodies only, so the single
    highest-blast-radius spelling of this anti-pattern -- a module-level
    ``except Exception: pytest.skip(..., allow_module_level=True)`` around an
    import -- passed unseen. Four of them were live in the suite.
    """
    # First against synthesised source, so the proof does not depend on those
    # four modules staying broken.
    synthetic = ast.parse(textwrap.dedent("""
        import pytest
        try:
            import spacr.gui_elements as ge
        except Exception as e:
            pytest.skip(f"unavailable: {e}", allow_module_level=True)

        def test_something():
            try:
                import spacr.nowhere
            except Exception:
                pytest.skip("also broad, but inside a function")
    """))
    scopes = dict(_scan_broad_skips(synthetic))
    assert MODULE_SCOPE in scopes, (
        "a module-level `except Exception: pytest.skip(...)` is not being "
        "seen; the module-scope walk in _scan_broad_skips is broken")
    assert "test_something" in scopes, "the in-function walk regressed"

    module_level = [(f, ln) for f, scope, ln in _broad_skip_sites()
                    if scope == MODULE_SCOPE]
    assert module_level, (
        "no module-level broad-skip found anywhere in tests/. Either they "
        "were all fixed -- in which case drop their MODULE_SCOPE entries from "
        "BROAD_SKIP_RATCHET and this test -- or the module-scope walk in "
        "_broad_skip_sites has stopped working.")
    # And every one of them is accounted for, by file.
    listed = {f for f, scopes in BROAD_SKIP_RATCHET.items()
              if MODULE_SCOPE in scopes}
    assert {f for f, _ in module_level} == listed


# ---------------------------------------------------------------------------
# Rule 3 — no machine-specific absolute paths
# ---------------------------------------------------------------------------

#: A user home directory. Anything under one exists on exactly one machine, so
#: a test gated on it is green-by-default everywhere else. ``/tmp`` and paths
#: inside the repo are fine; those exist wherever the suite runs.
_USER_HOME_PATH = re.compile(r"^/(home|Users)/[^/\s]+/")

#: (module, reason) pairs allowed to mention a user-home path.
USER_HOME_PATH_RATCHET = {
    # Asserts the string is ABSENT from spacr's shipped defaults -- the guard
    # against exactly the problem this rule is about.
    "test_settings_portable_defaults.py",
    # A synthetic path whose point is the space in the directory name (URL
    # quoting); never touched on disk.
    "qt/test_space_theme.py",
    # DEBT: the four @slow E2E tests default to
    # /home/carruthers/datasets/claude/{plate1,settings} and auto-skip
    # everywhere else, so they report green on every machine but one. They
    # already honour SPACR_E2E_DATA / SPACR_E2E_SETTINGS; the fix is to drop
    # the hard-coded defaults so the env vars are REQUIRED, making the tests
    # explicitly opt-in instead of silently skipped.
    "test_e2e_real_dataset.py",
}


def _string_constants(tree):
    """Every string literal in a module except the docstrings.

    Docstrings are prose -- a usage example may legitimately show
    ``/data/plate01`` -- while a literal in code is something the test uses.
    """
    docstring_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                docstring_nodes.add(id(node.body[0].value))
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docstring_nodes]


def test_no_user_home_paths_in_the_suite():
    """A path under someone's home directory is not a portable precondition."""
    offenders = []
    for path in _test_modules():
        rel = _rel(path)
        if rel in USER_HOME_PATH_RATCHET:
            continue
        for node in _string_constants(_parse(path)):
            for line in node.value.splitlines():
                if _USER_HOME_PATH.match(line.strip()):
                    offenders.append((rel, node.lineno, line.strip()[:70]))
                    break
    assert not offenders, (
        "these tests hard-code a path under a user's home directory:\n" +
        "\n".join(f"  {f}:{ln}  {t}" for f, ln, t in offenders) +
        "\n\nUse tmp_path, a fixture, or an environment variable. A test whose "
        "precondition only exists on one machine reports green everywhere else "
        "while running nothing."
    )


def _conftests():
    """Every conftest.py under tests/, nearest-the-root first.

    ``tests/conftest.py`` is not the only one: ``tests/qt/conftest.py`` is
    loaded for every Qt test in the suite and was never checked, so a stray
    absolute path there would have poisoned the whole Qt directory in silence.
    Any conftest added later is picked up automatically.
    """
    return sorted(TESTS_DIR.rglob("conftest.py"))


def test_conftest_hard_codes_no_absolute_path_at_all():
    """A conftest is loaded for EVERY test under it, so one stray path
    poisons the lot."""
    conftests = _conftests()
    assert {_rel(p) for p in conftests} >= {"conftest.py", "qt/conftest.py"}, (
        "the suite's known conftests are no longer being found; the glob in "
        "_conftests() has stopped matching")
    offenders = []
    for path in conftests:
        for node in _string_constants(_parse(path)):
            for line in node.value.splitlines():
                text = line.strip()
                if text.startswith("/") and len(text) > 1 \
                        and not text.startswith("/tmp"):
                    offenders.append((_rel(path), node.lineno, text[:70]))
    assert not offenders, (
        f"conftest(s) hard-code absolute path(s): {offenders}. Build "
        f"paths from tmp_path / tmp_path_factory or from the repo root.")


def test_the_e2e_dataset_paths_are_overridable_by_environment():
    """The one module on the path ratchet must at least honour env vars.

    This is what keeps the debt bounded: whatever the built-in default is, a
    second machine has to be able to point the module somewhere real without
    editing it.
    """
    source = (TESTS_DIR / "test_e2e_real_dataset.py").read_text(encoding="utf-8")
    assert "SPACR_E2E_DATA" in source
    assert "SPACR_E2E_SETTINGS" in source
    assert "os.environ.get" in source, (
        "the hard-coded dataset path is not overridable, so the module can "
        "only ever run on the machine it was written on")


# ---------------------------------------------------------------------------
# Rule 4 — the shared Cellpose mock contract actually rejects the bug
# ---------------------------------------------------------------------------
#
# tests/conftest.check_cellpose_eval_call is the guard the Cellpose mocks in
# this suite delegate to. A guard nobody tests is another way to be green for
# nothing, so it is exercised here against the exact call that survived
# fifteen tests and raised on every real run.


def test_the_cellpose_mock_contract_rejects_the_hardcoded_axis_3():
    """channel_axis=3 on a channels-last (H, W, C) image is the production bug.

    ``cellpose.transforms.convert_image`` indexes ``x.shape[channel_axis]``,
    and a 3-D array has no axis 3.
    """
    image = np.zeros((16, 16, 3), dtype=np.uint16)
    with pytest.raises(IndexError):
        check_cellpose_eval_call([image], 3)


def test_the_cellpose_mock_contract_rejects_an_axis_on_a_2d_image():
    """The other half of the same bug: a greyscale image takes no axis."""
    image = np.zeros((16, 16), dtype=np.uint16)
    with pytest.raises(ValueError, match="2D image"):
        check_cellpose_eval_call([image], -1)


def test_the_cellpose_mock_contract_accepts_what_spacr_actually_passes():
    """-1 on a channels-last stack, None on a 2-D image: both legal."""
    stack = np.zeros((16, 16, 2), dtype=np.uint16)
    converted = check_cellpose_eval_call([stack, stack], -1)
    assert len(converted) == 2
    # Cellpose pads to 3 channels; the spatial dims are untouched.
    assert converted[0].shape == (16, 16, 3)

    grey = np.zeros((16, 16), dtype=np.uint16)
    assert check_cellpose_eval_call(grey, None)[0].shape == (16, 16, 3)


def test_the_cellpose_mock_contract_notices_a_missing_channel_axis():
    """A mock that defaults the axis away is how channel_axis=3 got through."""
    stack = np.zeros((16, 16, 3), dtype=np.uint16)
    with pytest.raises(AssertionError, match="without channel_axis"):
        check_cellpose_eval_call([stack], MISSING_CHANNEL_AXIS)
    # ...and the sites that deliberately let Cellpose auto-detect opt out.
    check_cellpose_eval_call([stack], MISSING_CHANNEL_AXIS,
                             require_channel_axis=False)


#: CellposeModel doubles whose ``eval`` still absorbs ``channel_axis`` into
#: ``**kwargs``, and how many such ``eval`` methods each class holds. Seeded
#: from the modules outside the scope of the change that introduced this
#: contract; the list may only shrink. The fix is three lines per mock: name
#: the parameter, pass it to ``check_cellpose_eval_call``, and put the value
#: back into whatever the test records.
#:
#: The counts exist because a (file, class) key is not unique: a module can
#: define two classes with the same name in two different fixtures, and
#: ``test_cov_object_cellpose_masks.py`` defines ``_M`` twice.
#:
#: Six of these entries were invisible until the candidate filter stopped
#: reading class names. ``_M``, ``_RecordingCP`` and ``_FakeCP`` contain
#: neither "cellpose" nor "model", so the old ``if "cellpose" not in name and
#: "model" not in name: continue`` walked straight past them -- while the
#: docstring claimed the rule recognised a double by its method shape.
CELLPOSE_MOCK_RATCHET = {
    ("qt/test_annotate_worker_lifecycle.py", "FakeModel"): 1,
    ("qt/test_cpsam_diameter.py", "_RecordingCellposeModel"): 1,
    ("qt/test_live_preview_coverage.py", "_FakeCellposeModel"): 1,
    ("test_cov_object_cellpose_masks.py", "_M"): 2,
    ("test_cov_object_masks_sam.py", "_M"): 1,
    ("test_cov_object_preprocess_segment.py", "_RecordingCP"): 1,
    ("test_cov_submodules_cellpose_apply.py", "_FakeCellposeModel"): 1,
    ("test_cov_submodules_cellpose_train_test.py", "_FakeCellposeModel"): 1,
    ("test_coverage_fill_measure_object.py", "_FakeCP"): 1,
    ("test_model_compare.py", "StubModel"): 1,
    ("test_object_tstack_wiring.py", "_M"): 1,
    ("test_object_tstack_wiring.py", "_Model"): 1,
    ("test_spacrops_store_and_features.py", "CellposeModel"): 1,
}

#: Total offending ``eval`` methods above, not keys.
CELLPOSE_MOCK_CEILING = 14


#: Sentinel for "this default is not a python literal" (a Name like
#: ``MISSING_CHANNEL_AXIS``), which is exactly what a compliant mock uses.
_NOT_A_LITERAL = object()


def _param_default(args, name):
    """``(has_default, default_node)`` for parameter ``name`` of ``args``."""
    positional = args.posonlyargs + args.args
    names = [a.arg for a in positional]
    if name in names:
        offset = len(names) - len(args.defaults)
        index = names.index(name)
        if index >= offset:
            return True, args.defaults[index - offset]
        return False, None
    for kwarg, default in zip(args.kwonlyargs, args.kw_defaults):
        if kwarg.arg == name:
            return default is not None, default
    return False, None


def _channel_axis_complaint(fn):
    """Why ``fn`` (an ``eval`` method) fails the channel_axis contract, or None.

    Three ways to fail, all of which leave the mock unable to tell a working
    call from the ``channel_axis=3`` that raised on every real run:

    1. the parameter is not named at all, so ``**kwargs`` eats it;
    2. it is named but DEFAULTED to a value a caller could legally pass
       (``None``, or an axis index). Then "the caller omitted it" and "the
       caller passed it" are the same state, which is exactly the hole
       ``MISSING_CHANNEL_AXIS`` exists to close;
    3. it is named and never read. ``def eval(self, x, channel_axis=None,
       **kwargs)`` that ignores the value satisfies a rule that only checks
       the signature -- proved against the previous version of this rule --
       and validates nothing.
    """
    args = fn.args
    named = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
    if "channel_axis" not in named:
        return "swallows channel_axis into **kwargs"
    has_default, default = _param_default(args, "channel_axis")
    if has_default:
        try:
            # literal_eval, not `isinstance(default, ast.Constant)`: `-1` is a
            # UnaryOp(USub, Constant(1)), and -1 is the single most important
            # legal axis in this codebase to reject as a default.
            literal = ast.literal_eval(default)
        except (ValueError, SyntaxError, TypeError):
            literal = _NOT_A_LITERAL
        if literal is None or isinstance(literal, (int, float)):
            return (f"defaults channel_axis to {ast.unparse(default)}, which a "
                    f"real caller could pass -- use a sentinel like "
                    f"MISSING_CHANNEL_AXIS")
    used = any(isinstance(n, ast.Name) and n.id == "channel_axis"
               and isinstance(n.ctx, ast.Load)
               for n in ast.walk(fn))
    if not used:
        return "names channel_axis and never reads it"
    return None


def _cellpose_mock_offenders():
    """``(file, class, lineno, complaint)`` per non-compliant ``eval``.

    A candidate is any method named ``eval`` that takes ``**kwargs`` -- the
    method SHAPE, with no reference to the class's name. That is what the
    rule's docstring has always promised and what it did not do.
    """
    offenders = []
    for path in _test_modules():
        tree = _parse(path)
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            for fn in [n for n in cls.body
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
                if fn.name != "eval" or fn.args.kwarg is None:
                    continue      # no **kwargs: nothing is being swallowed
                complaint = _channel_axis_complaint(fn)
                if complaint:
                    offenders.append((_rel(path), cls.name, fn.lineno,
                                      complaint))
    return offenders


def test_the_cellpose_mocks_do_not_swallow_channel_axis():
    """Every CellposeModel stand-in names ``channel_axis`` on ``eval``,
    defaults it to a sentinel, and actually reads it.

    A mock spelled ``def eval(self, x, **kwargs)`` accepts every argument
    including the illegal ones, which is the mechanism -- not the symptom --
    behind the channel_axis=3 escape. Recognising the double by its method
    shape rather than its class name keeps this honest as new mocks appear.
    """
    found = {}
    for f, cls, lineno, complaint in _cellpose_mock_offenders():
        found.setdefault((f, cls), []).append((lineno, complaint))
    over_budget = []
    for key, hits in sorted(found.items()):
        allowed = CELLPOSE_MOCK_RATCHET.get(key, 0)
        if len(hits) > allowed:
            over_budget.append((key, sorted(hits), allowed))
    assert not over_budget, (
        "these CellposeModel doubles do not honour the channel_axis "
        "contract:\n" +
        "\n".join(f"  {f}: {c}.eval() at line(s) "
                  + ", ".join(f"{ln} ({why})" for ln, why in hits)
                  + f" -- ratchet allows {allowed}"
                  for (f, c), hits, allowed in over_budget) +
        "\n\nName it: `def eval(self, x, channel_axis=MISSING_CHANNEL_AXIS, "
        "**kwargs)` and hand the pair to check_cellpose_eval_call (see "
        "tests/conftest.py). A mock that accepts any axis cannot tell a "
        "working call from the one that crashed every real run."
    )


def test_the_cellpose_mock_ratchet_only_shrinks():
    total = sum(CELLPOSE_MOCK_RATCHET.values())
    assert total <= CELLPOSE_MOCK_CEILING, (
        f"CELLPOSE_MOCK_RATCHET grew to {total} eval methods (ceiling "
        f"{CELLPOSE_MOCK_CEILING}). Entries come off this list, never on.")
    missing = sorted({f for f, _ in CELLPOSE_MOCK_RATCHET
                      if not (TESTS_DIR / f).is_file()})
    assert not missing, (
        f"CELLPOSE_MOCK_RATCHET names modules that no longer exist: {missing}")


def test_the_cellpose_mock_rule_rejects_a_declared_but_ignored_axis():
    """The rule's own blind spots, pinned against synthesised mocks.

    Written as source rather than as live classes on purpose: adding a
    deliberately-broken CellposeModel double to the suite would be a mock
    other tests could pick up.
    """
    def complaint(src):
        cls = ast.parse(textwrap.dedent(src)).body[0]
        return _channel_axis_complaint(cls.body[0])

    # 1. Not named at all -- the original rule caught this one.
    assert "swallows" in complaint("""
        class _Double:
            def eval(self, x, **kwargs):
                return [], None
    """)
    # 2. Named, defaulted to a legal value, and ignored. This PASSED the
    #    previous rule, which only checked that the name appeared.
    assert "defaults channel_axis" in complaint("""
        class _Double:
            def eval(self, x, channel_axis=None, **kwargs):
                return [], None
    """)
    assert "defaults channel_axis" in complaint("""
        class _Double:
            def eval(self, x, channel_axis=-1, **kwargs):
                return [], None
    """)
    # 3. Named with a proper sentinel, but the value is never read.
    assert "never reads it" in complaint("""
        class _Double:
            def eval(self, x, channel_axis=MISSING_CHANNEL_AXIS, **kwargs):
                return [], None
    """)
    # ...and the shape the suite's compliant mocks actually use passes.
    assert complaint("""
        class _Double:
            def eval(self, x, channel_axis=MISSING_CHANNEL_AXIS, **kwargs):
                check_cellpose_eval_call(x, channel_axis)
                return [], None
    """) is None


def test_the_cellpose_mock_rule_does_not_read_class_names():
    """Six doubles were invisible because they are not called ``*Model``.

    ``_M``, ``_RecordingCP`` and ``_FakeCP`` are CellposeModel stand-ins by
    behaviour and by nothing else; a name filter cannot see them.
    """
    by_name = {cls for _, cls, _, _ in _cellpose_mock_offenders()}
    unnamed = {c for c in by_name
               if "cellpose" not in c.lower() and "model" not in c.lower()}
    assert unnamed, (
        "no name-invisible Cellpose double found. Either they were all fixed "
        "-- in which case trim CELLPOSE_MOCK_RATCHET and this test -- or the "
        "candidate filter has gone back to matching on class names.")
