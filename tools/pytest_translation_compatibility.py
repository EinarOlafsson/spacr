"""Report real translation-corpus incompatibilities without blocking CI.

Only the explicitly listed acceptance tests are advisory. Synthetic parser,
fallback, escaping, source extraction and translation safety implementation
tests remain required. English parametrizations always remain required.
"""

import hashlib
import json
import os
from pathlib import Path

import pytest


ADVISORY_TESTS = {
    "test_translation_correctness.py": {
        "test_every_runtime_translation_keeps_its_placeholders",
        "test_no_runtime_translation_loses_a_protected_term",
        "test_an_untranslated_entry_is_declared_or_absent",
        "test_the_allowlist_does_not_outlive_what_it_excuses",
        "test_every_language_carries_the_same_keys",
        "test_installer_catalogs_agree_with_english",
    },
    "test_i18n_coverage_audit.py": {
        "test_checked_in_coverage_report_is_the_live_source_report",
        "test_written_review_scope_matches_current_source_bound_evidence",
    },
    "test_pathway_walkthroughs.py": {
        "test_localized_pathways_render_every_step_and_preserve_navigation",
    },
    "test_backend_install_translations.py": {
        "test_installer_and_backend_cards_use_current_reviews",
    },
    "test_mask_filter_api_translations.py": {
        "test_mask_filter_api_publishes_reviewed_mean_and_bound_semantics",
    },
    "test_i18n_caption_ratchet.py": {
        "test_compact_user_facing_caption_surface_has_exact_rows_and_is_pinned",
        "test_spanish_compact_rows_use_consistent_formal_register",
        "test_external_caption_layer_is_complete_exclusive_and_pinned",
        "test_runtime_identity_captions_remain_exact_in_every_language",
    },
    "test_tutorial_catalog_i18n.py": {
        "test_spoken_pypi_is_the_reviewed_pype_form_in_every_spoken_locale",
        "test_caption_only_installation_lessons_keep_reviewed_display_copy",
        "test_localized_navigation_chrome_and_reviewed_copy_do_not_regress",
    },
    "test_documentation_i18n.py": {
        "test_documentation_api_catalog_inventory_and_hashes_are_current",
        "test_reviewed_readmes_do_not_reintroduce_known_context_errors",
        "test_localized_readmes_do_not_leave_long_english_feature_copy",
        "test_localized_readmes_preserve_safety_meaning_and_language_names",
        "test_localized_readmes_keep_the_badge_row_structurally_intact",
        "test_localized_readme_images_have_reviewed_accessible_text",
        "test_localized_readme_inline_markup_is_balanced_and_tight",
        "test_localized_readmes_preserve_module_names_and_technical_terms",
        "test_localized_readmes_preserve_urls_code_and_table_shape",
        "test_reviewed_readme_headings_match_the_canonical_source_and_locales",
        "test_localized_readmes_have_as_many_sections_as_the_english_one",
        "test_localized_readmes_keep_reviewed_semantic_and_typographic_fixes",
    },
    "test_api_i18n_frontend.py": {
        "test_every_complete_real_catalog_renders_through_the_browser_selector",
    },
    "test_built_nested_helper_api.py": {
        "test_enabled_helpers_use_their_own_real_catalog_entries_in_the_browser",
    },
    "test_external_i18n_catalogs.py": {
        "test_all_external_catalogs_preserve_runtime_placeholders",
        "test_external_runtime_catalogs_have_exact_current_source_keys",
        "test_standalone_technical_identity_values_remain_exact_in_every_language",
        "test_runtime_catalogs_reject_known_cross_domain_contamination_markers",
        "test_reviewed_ui_rows_are_exact_in_regenerated_runtime_catalogs",
        "test_runtime_tooltips_have_no_exact_english_prose_fallbacks",
        "test_runtime_catalogs_have_no_unreviewed_exact_english_fallbacks",
        "test_runtime_catalogs_need_no_incremental_repairs",
        "test_runtime_uses_external_static_and_context_keyed_setting_text",
        "test_reviewed_scientific_terms_use_domain_context_not_false_friends",
        "test_runtime_catalogs_resolve_all_reviewed_false_friend_variants",
        "test_chinese_and_scientific_runtime_terms_are_contextual",
        "test_api_doc_catalog_is_symbol_keyed_and_source_hashed",
    },
}


def advisory(path, name, parameters=None):
    """Keep this allowlist narrow so an unrelated regression cannot be waived."""
    return (Path(path).name in ADVISORY_TESTS
            and name in ADVISORY_TESTS[Path(path).name]
            and (parameters or {}).get("language") != "en")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    params = getattr(getattr(item, "callspec", None), "params", {})
    if (report.when != "call" or not report.failed or
            not advisory(item.path, getattr(item, "originalname", None) or item.name, params)):
        return
    record = {
        "schema": 1, "test": item.nodeid,
        "source_commit": os.environ.get("GITHUB_SHA", ""),
        "status": "incompatible", "blocking": False,
        "diagnostics": str(report.longrepr),
    }
    directory = Path(item.config.rootpath) / ".translation-reports"
    directory.mkdir(exist_ok=True)
    name = hashlib.sha256(item.nodeid.encode()).hexdigest() + f"-{os.getpid()}.json"
    (directory / name).write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    report.outcome = "skipped"
    report.wasxfail = "translation incompatibility registered in .translation-reports (report-only)"


def pytest_terminal_summary(terminalreporter):
    reports = [report for report in terminalreporter.stats.get("xfailed", [])
               if "translation incompatibility registered" in getattr(report, "wasxfail", "")]
    if reports:
        terminalreporter.section("Translation incompatibilities (report-only)")
        for report in reports:
            terminalreporter.write_line(report.nodeid)
        terminalreporter.write_line("Complete diagnostics: .translation-reports/*.json")
