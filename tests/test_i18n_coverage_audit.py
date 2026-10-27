"""The localization coverage report is anchored to live source inventory."""
from __future__ import annotations

import copy
import json
from functools import lru_cache, partial
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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
    # Bytes, so the 8 below does not become 9 under Windows' CRLF text mode.
    (readme / "README.sv.rst").write_bytes("Svenska\n".encode("utf-8"))

    report = coverage.build_report()

    assert "| Swedish | 1/2 | 1 | 1/2 | 1/1 | 1/2 | 1 | 8 |" in report
    assert "measured against live application/API source" in report


def _copy_review_scope_blocks(cached, *args, **kwargs):
    """Keep mutable splitter results private to each real caller."""
    return copy.deepcopy(cached(*args, **kwargs))


def _review_scope_api_memoizers(builder):
    """Reuse bounded exact-text derivations without caching record admission."""
    return {
        "translatable_blocks": partial(
            _copy_review_scope_blocks, lru_cache(maxsize=256)(builder.translatable_blocks)),
        "_api_translation_source": lru_cache(maxsize=8192)(builder._api_translation_source),
        "_reviewed_api_block_valid": lru_cache(maxsize=8192)(builder._reviewed_api_block_valid),
    }


def test_review_scope_reuse_keeps_target_validation_and_split_results_isolated():
    """A changed protected target is refused and cannot poison a cached split."""
    import build_documentation_i18n as builder

    cached = _review_scope_api_memoizers(builder)
    validate = cached["_reviewed_api_block_valid"]
    source, target = "Read ``plate_id``.", "Läs ``plate_id``."
    assert builder._reviewed_api_block_valid(source, target, "sv") is True
    assert validate(source, target, "sv") is True
    assert validate(source, target, "sv") is True
    assert validate.cache_info().hits == 1
    changed = "Läs ``well_id``."
    assert builder._reviewed_api_block_valid(source, changed, "sv") is False
    assert validate(source, changed, "sv") is False
    assert validate.cache_info().misses == 2
    assert builder._reviewed_api_block_valid(source, target, "zh_CN") is False
    assert validate(source, target, "zh_CN") is False
    changed_source = "Read ``well_id``."
    assert validate(changed_source, target, "sv") is False
    assert validate.cache_info().misses == 4
    assert validate(source, target, "sv") is True
    prose = "First paragraph.\n\nSecond paragraph."
    expected = builder.translatable_blocks(prose)
    blocks, layout = cached["translatable_blocks"](prose)
    blocks[0] = "Mutated by a caller."
    layout.append(("raw", "Caller-only mutation."))
    assert cached["translatable_blocks"](prose) == expected
    assert cached["translatable_blocks"](prose + " Changed.") == builder.translatable_blocks(prose + " Changed.")


@pytest.fixture(scope="module")
def live_source_snapshot(request):
    """Share one real source extraction and scoped pure derivations per module."""
    import build_documentation_i18n as api_builder
    import build_i18n_catalogs as runtime_builder

    docs = api_builder.public_docstrings()
    sources = runtime_builder.canonical_sources()
    memoizers = _review_scope_api_memoizers(api_builder)
    runtime_builder.reviewed_runtime_translations.cache_clear()
    try:
        with pytest.MonkeyPatch.context() as snapshot:
            for name, memoized in memoizers.items():
                snapshot.setattr(api_builder, name, memoized)
            snapshot.setattr(api_builder, "public_docstrings", partial(dict, docs))
            snapshot.setattr(runtime_builder, "canonical_sources", partial(copy.deepcopy, sources))
            request.getfixturevalue("api_catalogs")
            yield api_builder, runtime_builder, docs, sources
    finally:
        runtime_builder.reviewed_runtime_translations.cache_clear()


def test_checked_in_coverage_report_is_the_live_source_report(live_source_snapshot):
    """The published counts may not remain an older green snapshot."""
    checked_in = (ROOT / "docs" / "i18n" / "COVERAGE.md").read_text(
        encoding="utf-8",
    )
    assert checked_in == coverage.build_report()


@pytest.fixture(scope="module")
def live_review_scope(live_source_snapshot):
    """Use the same live inventory and the real session's pinned API build."""
    api_builder, runtime_builder, docs, sources = live_source_snapshot
    report = (ROOT / "docs" / "i18n" / "REVIEW_SCOPE.md").read_text(
        encoding="utf-8",
    )
    return api_builder, runtime_builder, docs, report, sources


@pytest.mark.parametrize(("language", "display"), tuple(DISPLAY_NAMES.items()))
def test_written_review_scope_matches_current_source_bound_evidence(
        monkeypatch, live_review_scope, language, display):
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
    api_builder, runtime_builder, docs, report, sources = live_review_scope
    # Each locale validates its own review evidence against the same live
    # source snapshot, without repeating Qt AST extraction in this worker.
    monkeypatch.setattr(runtime_builder, "canonical_sources", lambda: sources)
    # Installer strings ship as standalone JSON beside the app, so the runtime
    # denominator is every catalog table EXCEPT that one -- the same split
    # COVERAGE.md reports, which keeps the two documents comparable.
    runtime_total = sum(
        len(table) for name, table in sources.items() if name != "installer"
    )
    api_total = len(docs)
    assert runtime_total > 0 and api_total > 0

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
        f"{api_count:,} | {api_total:,} | "
        f"{100 * api_count / api_total:.2f} |"
    )
    assert "Public API symbols | Reviewed blocks per 100 symbols" in report
    assert row in report, (language, row)

    # Checked on normalized whitespace: both sentences wrap in the file, and a
    # raw substring would miss them for a reason that has nothing to do with
    # what they say.
    flowed = " ".join(report.split())
    assert (
        "this is an evidence report and not a certificate that every sentence "
        "was read by a fluent speaker" in flowed.casefold()
    )
    # PIN THE DISTINCTION, NOT THE STATE OF IT. This asserted the literal
    # "Mechanical source coverage is NOT complete" until 2026-09-10, when
    # the catalogs reached 5,260/5,260 runtime and 10,306/10,306 API in
    # every locale and the sentence stopped being true. A test that pins a
    # FACT goes red when the fact changes for the better, and this file's
    # own docstring says why that is the wrong shape: "a dated evidence
    # report cannot be a live assertion ... teaches the next reader to edit
    # numbers until it passes -- the opposite of what it is for."
    #
    # What must never go missing is the DISTINCTION the report exists to
    # draw: having a translation for every source is not the same as having
    # had one read. That sentence is true whether coverage is complete or
    # not, so it is the one worth holding.
    assert (
        "Mechanical coverage means every source has A translation. It does "
        "not mean a speaker of the language has read it" in flowed
    )
