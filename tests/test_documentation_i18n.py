"""Structural contracts for external API/docstring localization."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def test_docstring_reflow_preserves_rst_roles_fields_and_inline_code():
    from build_documentation_i18n import rebuild_document, translatable_blocks

    source = """Overview with :mod:`spacr.core` and ``an_inline_value``.

:func:`spacr.example` writes ``first, second,
third`` without changing the literal.

:param source: Input path.
:returns: A :class:`Result`.
"""
    blocks, layout = translatable_blocks(source)
    rebuilt = rebuild_document(layout, blocks)
    for value in (
        ":mod:`spacr.core`",
        ":func:`spacr.example`",
        "``an_inline_value``",
        "``first, second, third``",
        ":param source:",
        ":class:`Result`",
    ):
        assert value in rebuilt


def test_readme_language_picker_is_never_sent_through_translation():
    from build_documentation_i18n import rebuild_document, translatable_blocks

    picker = (
        "Languages: `English <README.rst>`_ · "
        "`Svenska <docs/i18n/readme/README.sv.rst>`_ ·\n"
        "`简体中文 <docs/i18n/readme/README.zh_CN.rst>`_\n\n"
        "Translate this sentence."
    )
    blocks, layout = translatable_blocks(picker)
    assert blocks == ["Translate this sentence."]
    assert rebuild_document(layout, blocks).startswith(
        "Languages: `English <README.rst>`_ · "
        "`Svenska <docs/i18n/readme/README.sv.rst>`_ ·\n"
        "`简体中文 <docs/i18n/readme/README.zh_CN.rst>`_"
    )


def test_github_summary_has_reviewed_domain_translations():
    from build_documentation_i18n import (
        REVIEWED_README_BLOCKS,
        REVIEWED_README_HEADINGS,
    )

    assert len(REVIEWED_README_BLOCKS) == 7
    for reviewed in REVIEWED_README_BLOCKS.values():
        assert set(reviewed) == {
            "sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr",
        }
    joined = {
        language: " ".join(block[language] for block in REVIEWED_README_BLOCKS.values())
        for language in next(iter(REVIEWED_README_BLOCKS.values()))
    }
    assert "CRISPR 筛选" in joined["zh_CN"]
    assert "criblages CRISPR" in joined["fr"]
    assert "CRISPR 스크리닝" in joined["ko"]
    assert "CRISPR-skim" in joined["is"]
    assert len(REVIEWED_README_HEADINGS) == 22
    for reviewed in REVIEWED_README_HEADINGS.values():
        assert set(reviewed) == {
            "sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr",
        }
    assert REVIEWED_README_HEADINGS["New in 1.5.0.0"]["hi"].endswith(
        "1.5.0.0 में नया"
    )


def test_translation_protection_has_no_nested_tokens_and_round_trips():
    from build_i18n_catalogs import _protect, _restore

    source = (
        "Each entry is {'name': <class>, 'where': [{'column': <feature>, "
        "'op': '>='}], :func:`spacr.core.run` keeps CUDA unchanged, and "
        "`the release <https://example.test/release>`_ remains a valid link."
    )
    protected, mapping = _protect(source)
    assert all("<x" not in value for value in mapping.values())
    assert _restore(protected, mapping) == source


def test_rejected_models_use_the_reviewed_permissive_replacement():
    from build_i18n_catalogs import MODEL_SPECS

    for language in ("zh_CN", "hi", "ko", "is"):
        model, _folder, license_name, _prefix = MODEL_SPECS[language]
        assert model == "facebook/m2m100_418M"
        assert license_name == "MIT"


def test_generation_loop_detection_rejects_repeated_labels():
    from build_i18n_catalogs import _looks_degenerate

    assert _looks_degenerate("Background", "背景" * 120, "zh_CN")
    assert _looks_degenerate("Run mode", "hamur hamur hamur hamur", "is")
    assert _looks_degenerate("A technical paragraph. " * 8, "traduction " * 80, "fr")
    assert not _looks_degenerate("Background", "背景", "zh_CN")


def test_incremental_api_generation_reuses_only_current_nonblank_entries(
    monkeypatch, tmp_path,
):
    import build_documentation_i18n as builder

    docs = {"spacr.example.current": "Current source.",
            "spacr.example.changed": "Changed source."}
    payload = {
        "symbols": {
            "spacr.example.current": {
                "source_sha256": hashlib.sha256(
                    docs["spacr.example.current"].encode()
                ).hexdigest(),
                "text": "Aktuell text.",
            },
            "spacr.example.changed": {
                "source_sha256": "stale",
                "text": "Gammal text.",
            },
        },
    }
    (tmp_path / "sv.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(builder, "API_DIR", tmp_path)

    assert builder.reusable_api_translations(docs, "sv") == {
        "spacr.example.current": "Aktuell text.",
    }


def test_reviewed_readmes_do_not_reintroduce_known_context_errors():
    readme_root = ROOT / "docs" / "i18n" / "readme"
    french = (readme_root / "README.fr.rst").read_text(encoding="utf-8")
    swedish = (readme_root / "README.sv.rst").read_text(encoding="utf-8")
    icelandic = (readme_root / "README.is.rst").read_text(encoding="utf-8")

    assert "l'criblage" not in french and "l’criblage" not in french
    assert "löpande ansökan" not in swedish
    assert "spaCR → Stillingar → Tungumál" in icelandic
