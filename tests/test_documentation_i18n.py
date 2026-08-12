"""Structural contracts for external API/docstring localization."""
from __future__ import annotations

import hashlib
import json
import re
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


def test_api_block_completeness_uses_exact_code_hash_allowlist():
    from build_documentation_i18n import (
        API_EXACT_BLOCK_SHA256_ALLOWLIST,
        _api_block_requires_translation,
        _source_hash,
    )
    from build_i18n_catalogs import _syntax_preserved

    prose = "Return the requested objects in deterministic order."
    code = "``(H, W, C)``."
    assert _api_block_requires_translation(prose)
    assert _source_hash(code) in API_EXACT_BLOCK_SHA256_ALLOWLIST
    assert not _api_block_requires_translation(code)
    assert _syntax_preserved(
        'Use "classifier_evaluation" with --dry-run.',
        'Use "classifier_evaluation" with --dry-run.',
    )
    assert not _syntax_preserved(
        'Use "classifier_evaluation" with --dry-run.',
        'Use " classifier_evaluation" with --dry-run.',
    )
    assert not _syntax_preserved(
        'Use "classifier_evaluation" with --dry-run.',
        'Use "classifier_evaluation" with --dry-run and "unexpected_mode".',
    )


def test_api_source_discovery_excludes_untracked_backup_icons():
    from build_documentation_i18n import public_docstrings

    assert not any("backup_icons" in key for key in public_docstrings())


def test_reviewed_readmes_do_not_reintroduce_known_context_errors():
    readme_root = ROOT / "docs" / "i18n" / "readme"
    french = (readme_root / "README.fr.rst").read_text(encoding="utf-8")
    swedish = (readme_root / "README.sv.rst").read_text(encoding="utf-8")
    icelandic = (readme_root / "README.is.rst").read_text(encoding="utf-8")

    assert "l'criblage" not in french and "l’criblage" not in french
    assert "löpande ansökan" not in swedish
    assert "spaCR → Stillingar → Tungumál" in icelandic


def test_localized_readmes_do_not_leave_long_english_feature_copy():
    """GitHub's feature table and surrounding guidance must be localized."""
    readme_root = ROOT / "docs" / "i18n" / "readme"
    localized = {
        path.stem.removeprefix("README."): path.read_text(encoding="utf-8")
        for path in readme_root.glob("README.*.rst")
    }
    assert set(localized) == {
        "sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr",
    }

    canonical = (ROOT / "README.rst").read_text(encoding="utf-8")
    table = canonical[
        canonical.index(".. list-table::"):
        canonical.index(".. |api-qt-app| replace::")
    ]
    descriptions = {
        line.strip()[2:]
        for line in table.splitlines()
        if line.startswith("     - ")
        and len(line.strip()[2:].split()) >= 6
    }
    assert len(descriptions) == 28

    long_prose_fragments = {
        "The installer downloads a private Python 3.12 runtime",
        "Runs are now identifiable",
        "Navigation, Preferences, AI and LIVE controls",
        "94 short animations explain what 143 visual settings",
        "Bug reports and focused feature requests",
        "The current development branch is source-available",
        "contains narrated, captioned walkthroughs",
        "segments cells, nuclei, pathogens and organelles with Cellpose",
        "In the evaluation screen, a confusion-matrix cell is a query",
    }
    table_labels = {
        "**Ten-language localization**",
        "**Localized contextual help**",
        "**Setting animation registry**",
        "**Visual setting animations**",
        "**Installation diagnosis**",
        "**Flat-field correction**",
        "**Object measurements**",
        "**Well and collision report**",
        "**Screen effect estimation**",
        "**Run provenance**",
    }
    forbidden = descriptions | long_prose_fragments | table_labels
    for language, text in localized.items():
        leftovers = sorted(fragment for fragment in forbidden if fragment in text)
        assert not leftovers, f"{language} retains English README copy: {leftovers}"


def test_localized_readmes_preserve_safety_meaning_and_language_names():
    readme_root = ROOT / "docs" / "i18n" / "readme"
    spanish = (readme_root / "README.es.rst").read_text(encoding="utf-8")
    hindi = (readme_root / "README.hi.rst").read_text(encoding="utf-8")
    korean = (readme_root / "README.ko.rst").read_text(encoding="utf-8")

    # The exporter rejects invented numbers; the old machine translation
    # reversed this safety guarantee.
    assert "se rechaza cualquier borrador" in spanish
    assert "no es rechazado" not in spanish
    assert "no se rechaza" not in spanish

    # Hindi is a language, not the Hindu religion.
    assert "हिन्दी" in hindi
    assert "हिंदू" not in hindi and "हिन्दू" not in hindi
    assert "힌디어" in korean
    assert "힌두교" not in korean

    # Common literal-translation failures in scientific/software context.
    for false_friend in ("la antorcha", "el gasoducto", "cara de agarre", "Open Daughth"):
        assert false_friend not in spanish


def test_localized_readmes_keep_the_badge_row_structurally_intact():
    expected = (
        "|Docs| |Tutorials| |PyPI| |Python| |Tests| |Qt| "
        "|Source| |Issues| |License| |DOI|"
    )
    for path in (ROOT / "docs" / "i18n" / "readme").glob("README.*.rst"):
        assert path.read_text(encoding="utf-8").splitlines()[0] == expected


def test_localized_readme_images_have_reviewed_accessible_text():
    expected_workflow_alt = {
        "de": "spaCR-Arbeitsablauf und Ausgabeorganisation",
        "es": "Flujo de trabajo y organización de resultados de spaCR",
        "fr": "Flux de travail spaCR et organisation des sorties",
        "hi": "spaCR कार्यप्रवाह और आउटपुट संगठन",
        "is": "Verkflæði spaCR og skipulag úttaks",
        "ko": "spaCR 작업 흐름 및 출력 구성",
        "pt": "Fluxo de trabalho e organização das saídas do spaCR",
        "sv": "spaCR:s arbetsflöde och struktur för utdata",
        "zh_CN": "spaCR 工作流程及输出结构",
    }
    readme_root = ROOT / "docs" / "i18n" / "readme"
    for language, workflow_alt in expected_workflow_alt.items():
        text = (readme_root / f"README.{language}.rst").read_text(
            encoding="utf-8"
        )
        alt_text = re.findall(r"(?m)^   :alt: (.+)$", text)
        assert len(alt_text) == 14
        assert alt_text[-1] == workflow_alt
        assert alt_text[-1] != "spaCR workflow and output organization"
        assert "Interactive tutorials" not in alt_text
        assert "Latest installers" not in alt_text


def test_localized_readme_inline_markup_is_balanced_and_tight():
    canonical = (ROOT / "README.rst").read_text(encoding="utf-8")
    for path in (ROOT / "docs" / "i18n" / "readme").glob("README.*.rst"):
        text = path.read_text(encoding="utf-8")
        for marker in ("**", "``"):
            assert text.count(marker) == canonical.count(marker)
            marked_text = text.split(marker)[1::2]
            assert all(value and value == value.strip() for value in marked_text)

        # A prior Portuguese translation added visible ``>`` characters after
        # links and inline literals. Real link-target brackets are removed first.
        without_link_targets = re.sub(r"<[^>\n]+>", "", text)
        assert ">" not in without_link_targets


def test_localized_readmes_preserve_module_names_and_technical_terms():
    expected_modules = [
        "Mask", "Measure", "Annotate", "Classify", "Map Barcodes", "Regression",
    ]
    protected_terms = {
        "torchvision", "btrack", "pylibCZIrw", "czifile", "Hugging Face",
        "Power / Design", "ComBat", "scanpy",
    }
    fallback_phrases = {
        "de": "nicht unterstützten Gebietsschemata",
        "es": "configuraciones regionales no compatibles",
        "fr": "paramètres régionaux non pris en charge",
        "hi": "असमर्थित लोकेल",
        "is": "Tungumál sem ekki eru studd",
        "ko": "지원되지 않는 로캘",
        "pt": "Localidades não compatíveis",
        "sv": "Språk som inte stöds",
        "zh_CN": "不支持的语言环境",
    }
    known_context_errors = {
        "de": {"Fackelvision"},
        "es": {"Anotate", "la antorcha", "el gasoducto", "cara de agarre"},
        "fr": {"Face de harnais", "Anotate"},
        "hi": {"चेहरे को हिलाना", "**मैप बारकोड**", "**ग्रेसेज**"},
        "is": {"kyndilssýn"},
        "ko": {"전체 미생물", "전원 / 디자인", "그래프 건축가"},
        "pt": {"pylibCZrw", "**Máscara**", "**Mapa códigos de barras**"},
        "sv": {"Huggande ansikte", "**Mäta**", "**Karta Streckkoder**"},
        "zh_CN": {"此分類上一篇", "印度语", "电源 / 设计", "图形建筑师"},
    }
    readme_root = ROOT / "docs" / "i18n" / "readme"
    for language, fallback_phrase in fallback_phrases.items():
        text = (readme_root / f"README.{language}.rst").read_text(
            encoding="utf-8"
        )
        module_lines = re.findall(r"(?m)^\*\*([^*\n]+)\*\* .+$", text)
        assert module_lines[:6] == expected_modules
        missing_terms = sorted(term for term in protected_terms if term not in text)
        assert not missing_terms, f"{language} changed protected terms: {missing_terms}"
        assert fallback_phrase in text
        provenance = next(line for line in text.splitlines() if "AnnData" in line)
        assert all(name in provenance for name in ("Mask", "Measure", "Classify"))
        assert not any(error in text for error in known_context_errors[language])


def test_localized_readmes_preserve_urls_code_and_table_shape():
    canonical = (ROOT / "README.rst").read_text(encoding="utf-8")
    canonical_urls = sorted(re.findall(r"https?://[^\s>`]+", canonical))
    code_pattern = re.compile(
        r"(?m)^\.\. code-block:: [^\n]+\n\n((?: {3}[^\n]*(?:\n|$))+)",
    )
    canonical_code = code_pattern.findall(canonical)
    for path in (ROOT / "docs" / "i18n" / "readme").glob("README.*.rst"):
        text = path.read_text(encoding="utf-8")
        assert sorted(re.findall(r"https?://[^\s>`]+", text)) == canonical_urls
        assert code_pattern.findall(text) == canonical_code
        assert len(re.findall(r"(?m)^   \* - ", text)) == 33
        for target in re.findall(r"<((?:\.\.?/)[^>#]+)(?:#[^>]*)?>`_", text):
            assert (path.parent / target).resolve().exists(), (path, target)


def test_reviewed_readme_headings_match_the_canonical_source_and_locales():
    from build_documentation_i18n import (
        REVIEWED_README_HEADINGS,
        translatable_blocks,
    )

    canonical = (ROOT / "README.rst").read_text(encoding="utf-8")
    source_blocks, _ = translatable_blocks(canonical)
    assert "Animated setting guidance" in REVIEWED_README_HEADINGS
    assert "Animated settings guidance" not in REVIEWED_README_HEADINGS
    for source, localized_headings in REVIEWED_README_HEADINGS.items():
        assert source_blocks.count(source) == 1
        for language, heading in localized_headings.items():
            localized = (
                ROOT / "docs" / "i18n" / "readme" / f"README.{language}.rst"
            ).read_text(encoding="utf-8")
            assert heading in localized, (source, language, heading)


def test_localized_readmes_keep_reviewed_semantic_and_typographic_fixes():
    readme_root = ROOT / "docs" / "i18n" / "readme"
    readmes = {
        language: (readme_root / f"README.{language}.rst").read_text(
            encoding="utf-8"
        )
        for language in ("de", "es", "hi", "ko", "pt", "sv", "zh_CN")
    }

    required = {
        "de": {"spaCR-Installationsverzeichnis", "Animierte Einstellungshilfe"},
        "es": {
            "coherencia de las dependencias",
            "La interfaz Tk heredada",
            "Guía animada de ajustes",
        },
        "hi": {
            "अनुक्रमण त्रुटि",
            "एनोटेटर सहमति",
            "गुणांकों",
            "संभावित परिणामों की सूची",
            "एक पूल्ड, छवि-आधारित CRISPR स्क्रीन",
            "ESCRT तंत्र के अपहरण",
            "स्वागत है",
        },
        "ko": {
            "그래도 열기",
            "풀드 이미지 기반 CRISPR 스크린",
            "*T. gondii*\\ 의 ESCRT 기능 탈취",
        },
        "pt": {"Guia animado de configurações"},
        "sv": {"Animerad hjälp för inställningar"},
        "zh_CN": {
            "Windows 10/11：下载",
            "macOS 11+（英特尔和苹果硅）：下载",
            "64 位 Linux：下载",
            "测试数据集：Hugging Face toxo_mito",
            "测序数据：NCBI BioProject",
            "请引用：",
        },
    }
    forbidden = {
        "de": {"spaCR Installationsverzeichnis"},
        "es": {"interfaz Tk legado", "Orientación de ajuste animado"},
        "hi": {
            "स्वाग योग्य",
            "sequencing error",
            "dropout",
            "segmentation",
            "annotator agreement",
            "data leakage",
            "batch correction",
            "coefficients",
            "hit list",
            "एक संयुक्त छवि-आधारित CRISPR",
            "एकीकृत छवि-आधारित CRISPR",
            "ESCRT उपइकाई",
            "ESCRT उप-इकाई",
            "ESCRT उप इकाई",
            "ESCRT उप-विवाद",
        },
        "ko": {
            "Open Anyway",
            "합성 이미지 기반",
            "통합 이미지 기반",
            "ESCRT 하위",
        },
        "pt": {"Orientação de cenário animado"},
        "zh_CN": {
            "Windows 10/11:下载",
            "Windows 10/11: 下载",
            "测试数据集:",
            "请引用:",
        },
    }
    for language, fragments in required.items():
        missing = sorted(
            fragment for fragment in fragments if fragment not in readmes[language]
        )
        assert not missing, f"{language} lacks reviewed wording: {missing}"
    for language, fragments in forbidden.items():
        leftovers = sorted(
            fragment for fragment in fragments if fragment in readmes[language]
        )
        assert not leftovers, f"{language} retains reviewed errors: {leftovers}"
