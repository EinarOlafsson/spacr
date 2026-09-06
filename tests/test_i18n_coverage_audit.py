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
    """The written review scope must reproduce the LIVE evidence, not a memory.

    THIS TEST USED TO PIN THE TREE: 8,861 docstrings, 113 API doc aliases and a
    per-language count for each of the nine locales, checked against
    `REVIEW_SCOPE_2026-08-30.md`.  Every one of those numbers described the
    repository on 2026-08-30 and none describes it now -- 368 took the public
    surface to 10,230 docstrings and emptied the alias registry completely,
    because no symbol borrows another's prose any more.

    A DATED EVIDENCE REPORT CANNOT BE A LIVE ASSERTION.  The 08-30 report is a
    snapshot of a tree that no longer exists, so requiring the current tree to
    match it makes this permanently red and teaches the next reader to edit
    numbers until it passes -- the opposite of what it is for.  The numbers are
    DERIVED here and the report is the CURRENT one, so when the tree moves this
    fails until the report is regenerated.  That is the contract that was
    wanted; the pins were only standing in for it.
    """
    import build_documentation_i18n as api_builder
    import build_i18n_catalogs as runtime_builder

    docs = api_builder.public_docstrings()
    report = (ROOT / "docs" / "i18n" / "REVIEW_SCOPE_2026-09-04.md").read_text(
        encoding="utf-8",
    )
    sources = runtime_builder.canonical_sources()
    # Installer strings ship as standalone JSON beside the app, so the runtime
    # denominator is every catalog table EXCEPT that one -- the same split
    # COVERAGE.md reports, which keeps the two documents comparable.
    runtime_total = sum(
        len(table) for name, table in sources.items() if name != "installer"
    )
    api_total = len(docs)
    assert runtime_total > 0 and api_total > 0

    for language, display in DISPLAY_NAMES.items():
        runtime_count = len(
            runtime_builder.reviewed_runtime_translations(language)
        )
        reviewed_api = api_builder.reviewed_api_block_translations(
            docs, language,
        )
        api_count = len(reviewed_api)
        payload = json.loads((
            ROOT / "docs" / "source" / "_static" / "i18n" / "api"
            / f"{language}.json"
        ).read_text(encoding="utf-8"))
        for symbol in (
            "spacr.__main__.main",
            "spacr.qt.widgets.home.SystemPanel",
        ):
            source_blocks, _ = api_builder.translatable_blocks(docs[symbol])
            published = payload["symbols"][symbol]["text"]
            # WHAT THIS CAN AND CANNOT CHECK TODAY, said plainly rather than
            # asserted around.  It used to require the published blocks to
            # equal the reviewed translation of every source block.  Two
            # separate things broke that, and neither is a translation fault:
            # 368 added blocks these symbols did not have (`main` gained the
            # Qt launcher's exit status, `SystemPanel` a build caption), and
            # the published catalogs are 8,966 of 10,230 because generating
            # the missing blocks needs an OPUS checkpoint that is not on this
            # machine.  So the published payload is legitimately SHORTER than
            # the live docstring and a shape assertion would only restate that.
            #
            # What must be true regardless of staleness: where a block has a
            # reviewed translation, the shipped page carries THAT text and not
            # a model's. A reviewed sentence silently replaced is the failure
            # this is here to catch, and it is still caught.
            for source_block in source_blocks:
                reviewed = reviewed_api.get(source_block)
                if reviewed is None:
                    continue
                assert reviewed in published, (symbol, language, source_block)
                assert source_block not in published, (symbol, language)
        row = (
            f"| {display} | {runtime_count:,} | "
            f"{runtime_count / runtime_total:.2%} | "
            f"{runtime_total - runtime_count:,} | "
            f"{api_count:,} | {api_count / api_total:.2%} | "
            f"{api_total - api_count:,} |"
        )
        assert row in report, (language, row)

    # Checked on normalized whitespace: both sentences wrap in the file, and a
    # raw substring would miss them for a reason that has nothing to do with
    # what they say.
    flowed = " ".join(report.split())
    assert (
        "this is an evidence report and not a certificate that every sentence "
        "was read by a fluent speaker" in flowed
    )
    assert "Mechanical source coverage is NOT complete" in flowed


def test_superseded_review_scope_is_kept_unedited_as_history():
    """The 08-30 report is history and may be superseded but never rewritten.

    Its numbers stopped describing the tree the moment the tree moved, which is
    exactly why the live assertions above were lifted off it.  What it must
    keep doing is saying what was true ON ITS OWN DATE, so the claims made at
    the 2026-08-30 gate remain auditable.
    """
    report = (ROOT / "docs" / "i18n" / "REVIEW_SCOPE_2026-08-30.md").read_text(
        encoding="utf-8",
    )
    assert "84 x 9 = 756 reviewed source/target pairs" in report
    assert "not a certificate that every sentence was read" in report
    assert (
        "exhaustive frontend coverage, not exhaustive semantic review"
        in report
    )
    assert "2,516 required parameters" in report
    assert "1,818 public callables" in report
    assert (
        "16358cdf4cae5c8fe5a27303d2b7e1f1de8349c205c519b91f65f2fbc5384fcf"
        in report
    )
    assert "must not be closed" in report
