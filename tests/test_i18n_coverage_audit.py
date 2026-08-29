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
