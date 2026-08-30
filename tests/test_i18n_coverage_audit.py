"""The localization coverage report is anchored to live source inventory."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import audit_i18n_coverage as coverage  # noqa: E402

REVIEWED_RUNTIME_COUNTS = {
    "sv": 78, "de": 48, "es": 55, "zh_CN": 202, "pt": 59,
    "hi": 67, "ko": 183, "is": 73, "fr": 56,
}
REVIEWED_API_BLOCK_COUNTS = {
    "sv": 180, "de": 122, "es": 16, "zh_CN": 140, "pt": 128,
    "hi": 74, "ko": 106, "is": 588, "fr": 106,
}
DISPLAY_NAMES = {
    "sv": "Swedish", "de": "German", "es": "Spanish",
    "zh_CN": "Simplified Chinese", "pt": "Portuguese", "hi": "Hindi",
    "ko": "Korean", "is": "Icelandic", "fr": "French",
}


def test_report_ignores_stale_generated_english_manifests(
    tmp_path, monkeypatch,
):
    """English catalogs cannot ratify target catalogs stale in the same way."""
    live_runtime = {
        "SETTING_LABELS": {
            "current": "Current label",
            "edited": "Edited live label",
        },
        "SETTING_TOOLTIPS": {},
        "CATEGORY_HELP": {},
        "UI": {},
        "MODULE_SUMMARIES": {},
    }
    live_api = {
        "spacr.current": "Current API prose.",
        "spacr.edited": "Edited live API prose.",
    }
    target_module = SimpleNamespace(
        SETTING_LABELS={
            "current": "Aktuell etikett",
            "edited": "Gammal etikett",
            "orphan": "Föräldralös",
        },
        SETTING_TOOLTIPS={},
        CATEGORY_HELP={},
        UI={},
        MODULE_SUMMARIES={},
        SOURCE_HASHES={
            ("SETTING_LABELS", "current"):
                coverage._source_hash("Current label"),
            ("SETTING_LABELS", "edited"):
                coverage._source_hash("Old label"),
            ("SETTING_LABELS", "orphan"):
                coverage._source_hash("Orphaned label"),
        },
    )

    def catalog_import(name):
        assert name != "spacr.qt.i18n_catalogs.en"
        assert name == "spacr.qt.i18n_catalogs.sv"
        return target_module

    monkeypatch.setattr(coverage, "ROOT", tmp_path)
    monkeypatch.setattr(coverage, "LANGUAGES", ("sv",))
    monkeypatch.setattr(coverage, "DISPLAY", {"sv": "Swedish"})
    monkeypatch.setattr(coverage, "import_module", catalog_import)
    monkeypatch.setattr(
        coverage, "_live_runtime_tables", lambda: live_runtime,
    )
    monkeypatch.setattr(
        coverage, "_live_api_docstrings", lambda: live_api,
    )

    packaging = tmp_path / "packaging" / "i18n"
    packaging.mkdir(parents=True)
    (packaging / "en.json").write_text(
        json.dumps({"install": "Install"}), encoding="utf-8",
    )
    (packaging / "sv.json").write_text(
        json.dumps({"install": "Installera"}), encoding="utf-8",
    )
    api_root = tmp_path / "docs" / "source" / "_static" / "i18n" / "api"
    api_root.mkdir(parents=True)
    # This is intentionally invalid: a live-source report must never read it.
    (api_root / "en.json").write_text("{stale", encoding="utf-8")
    (api_root / "sv.json").write_text(json.dumps({
        "symbols": {
            "spacr.current": {
                "source_sha256": coverage._source_hash(
                    "Current API prose."
                ),
                "text": "Aktuell API-prosa.",
            },
            "spacr.edited": {
                "source_sha256": coverage._source_hash("Old API prose."),
                "text": "Gammal API-prosa.",
            },
            "spacr.orphan": {
                "source_sha256": coverage._source_hash(
                    "Orphaned API prose."
                ),
                "text": "Föräldralös API-prosa.",
            },
        },
    }), encoding="utf-8")
    readme = tmp_path / "docs" / "i18n" / "readme"
    readme.mkdir(parents=True)
    (readme / "README.sv.rst").write_text("Svenska\n", encoding="utf-8")

    report = coverage.build_report()

    assert "| Swedish | 1/2 | 1 | 1/2 | 1/1 | 1/2 | 1 | 8 |" in report
    assert "measured against live application/API source" in report


def test_checked_in_coverage_report_is_the_live_source_report():
    """The published counts may not remain an older green snapshot."""
    checked_in = (ROOT / "docs" / "i18n" / "COVERAGE.md").read_text(
        encoding="utf-8",
    )
    assert checked_in == coverage.build_report()


def test_written_review_scope_matches_current_source_bound_evidence():
    """Keep the honest reviewed subset and arithmetic remainder reproducible."""
    import build_documentation_i18n as api_builder
    import build_i18n_catalogs as runtime_builder

    docs = api_builder.public_docstrings()
    report = (ROOT / "docs" / "i18n" / "REVIEW_SCOPE_2026-08-30.md").read_text(
        encoding="utf-8",
    )
    assert len(docs) == 8_899
    for language, runtime_expected in REVIEWED_RUNTIME_COUNTS.items():
        api_expected = REVIEWED_API_BLOCK_COUNTS[language]
        reviewed_api = api_builder.reviewed_api_block_translations(
            docs, language,
        )
        assert len(
            runtime_builder.reviewed_runtime_translations(language)
        ) == runtime_expected
        assert len(reviewed_api) == api_expected
        source_blocks, _ = api_builder.translatable_blocks(
            docs["spacr.__main__.main"]
        )
        payload = json.loads((
            ROOT / "docs" / "source" / "_static" / "i18n" / "api"
            / f"{language}.json"
        ).read_text(encoding="utf-8"))
        translated_blocks, _ = api_builder.translatable_blocks(
            payload["symbols"]["spacr.__main__.main"]["text"]
        )
        assert translated_blocks == [reviewed_api[block] for block in source_blocks]
        row = (
            f"| {DISPLAY_NAMES[language]} | {runtime_expected:,} | "
            f"{runtime_expected / 4_960:.2%} | {4_960 - runtime_expected:,} | "
            f"{api_expected:,} | {api_expected / 8_899:.2%} | "
            f"{8_899 - api_expected:,} |"
        )
        assert row in report

    assert "84 x 9 = 756 reviewed source/target pairs" in report
    assert "not a certificate that every sentence was read" in report
    assert "must not be closed" in report
