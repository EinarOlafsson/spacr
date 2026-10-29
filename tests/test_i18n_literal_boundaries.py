"""Quoted API literals remain visible beside target-language grammar."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from build_i18n_catalogs import _contextualize, _syntax_preserved  # noqa: E402


@pytest.mark.parametrize("language", ["sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr"])
@pytest.mark.parametrize("source", ["ER (Y)", "DNA (B)"])
def test_subcell_colour_marker_is_scientific_identity(language, source):
    import build_i18n_catalogs as builder

    assert builder._reviewed_translation(source, language) == source
    assert builder._contextualize(source, language, source) == source
    assert not builder._translation_rejection_reasons(source, source, language)


def test_scientific_marker_identity_does_not_admit_copied_mapping_instructions():
    import build_i18n_catalogs as builder

    source = "Choose four distinct crop channels in SubCell's official order. Stain identity is not guessed from channel position."
    assert "exact" in builder._translation_rejection_reasons(source, source, "de", force=True)


@pytest.mark.parametrize("source", [
    "Open the organism guide from Home.",
    "Read the user guide before running the analysis.",
    "Consult the installation guide.",
    "The organism guide describes guide counts and guide filtering.",
])
@pytest.mark.parametrize("language, target", [
    ("zh_CN", "请阅读指南。"),
    ("de", "Lesen Sie den Leitfaden."),
])
def test_documentation_guide_does_not_become_guide_rna(source, language, target):
    assert _contextualize(target, language, source) == target


@pytest.mark.parametrize("language, target, corrected", [
    ("zh_CN", "比较指南计数。", "比较引导 RNA计数。"),
    ("de", "Vergleichen Sie die Leitfäden.", "Vergleichen Sie die Guide-RNAs."),
])
def test_molecular_guide_false_friend_still_gets_corrected(language, target, corrected):
    assert _contextualize(target, language, "Compare guide counts.") == corrected


def test_settings_dictionary_prose_is_not_a_parameter_declaration():
    source = "Python settings dictionary: iterable sweep values and max_workers."
    target = "Python-Einstellungswörterbuch: iterierbare Rasterwerte und max_workers."
    assert _syntax_preserved(source, target)
    assert not _syntax_preserved(source, target.replace('max_workers', 'max_processes'))


def test_actual_dictionary_parameter_type_stays_literal():
    source = "dictionary: iterable Values to inspect."
    assert _syntax_preserved(source, "dictionary: iterable Zu prüfende Werte.")
    assert not _syntax_preserved(source, "dictionary: list Zu prüfende Werte.")
    assert not _syntax_preserved(source, "Wörterbuch: iterierbar Zu prüfende Werte.")


def test_tiff_file_plural_keeps_format_without_imposing_english_grammar():
    assert _syntax_preserved('Import TIFFs.', 'Importez des fichiers TIFF.')
    assert not _syntax_preserved('Import TIFFs.', 'Importez des fichiers PNG.')
    assert not _syntax_preserved("Choose 'TIFFs'.", "Choisissez 'TIFF'.")
    assert not _syntax_preserved('Open data.TIFFs.', 'Ouvrez data.TIFF.')


def test_korean_particle_may_follow_an_exact_quoted_literal():
    source = "Use 'load_images' or 'stream_images'."
    translated = "'load_images'로 읽고 'stream_images'에서 자릅니다."

    assert _syntax_preserved(source, translated)
    assert not _syntax_preserved(
        source, translated.replace("'load_images'", "'load_image'")
    )


def test_english_apostrophes_do_not_become_quoted_api_literals():
    source = "Don't replace the user's selected source."
    translated = "사용자가 선택한 소스를 바꾸지 않습니다."

    assert _syntax_preserved(source, translated)


def test_comma_separated_option_values_survive_context_repairs():
    source = "The remaining text is left as is. Default 'above,left,below'."
    translated = "O texto restante é left como está. Padrão 'above,left,below'."
    result = _contextualize(translated, "pt", source)
    assert result == "O texto restante é deixado como está. Padrão 'above,left,below'."
    assert _syntax_preserved(source, result)
    assert not _syntax_preserved(source, result.replace("'above,left,below'", "'above,deixado,below'"))


def test_comma_separated_options_are_literals_with_either_quote_style():
    source = 'Choose "above,left,below" or \'left,below\'.'
    translated = 'Escolha "above,left,below" ou \'left,below\'.'
    assert _syntax_preserved(source, translated)
    assert _contextualize(translated, "pt", source) == translated
    assert not _syntax_preserved(source, translated.replace("left,below", "below,left"))


def test_spanish_cleanup_preserves_verbatim_citation_and_copyright(monkeypatch):
    import build_i18n_catalogs as builder
    monkeypatch.setattr(builder, "_reviewed_translation", lambda *_: None)
    title = "'Spotiflow: accurate and efficient spot detection'"
    notice = "(Copyright 2023 Albert Dominguez Mantes and Martin Weigert)"
    source = f"Published results: {title}, doi:10.1038/s41592-025-02662-x. {notice}"
    target = f"Resultados publicados: {title}, doi:10.1038/s41592-025-02662-x. {notice} and otros."
    expected = target.removesuffix("and otros.") + "y otros."
    assert builder._contextualize(target, "es", source) == expected
    assert builder._contextualize(expected, "es", source) == expected


def test_attribution_protection_is_bound_to_the_english_source(monkeypatch):
    import build_i18n_catalogs as builder
    monkeypatch.setattr(builder, "_reviewed_translation", lambda *_: None)
    target = "'accurate and efficient spot detection in microscopy'"
    assert builder._contextualize(target, "es", "Compare accuracy and efficiency.") == "'accurate y efficient spot detection in microscopy'"


@pytest.mark.parametrize("citation", ["arXiv:2408.00714", "ARXIV:2408.00714v2"])
def test_spanish_cleanup_preserves_verbatim_arxiv_title(monkeypatch, citation):
    import build_i18n_catalogs as builder
    monkeypatch.setattr(builder, "_reviewed_translation", lambda *_: None)
    title = "'SAM 2: Segment Anything in Images and Videos'"
    source = f"Published results: Ravi et al., {title}, ICLR 2025 ({citation})."
    target = f"Resultados publicados: Ravi et al., {title}, ICLR 2025 ({citation}), and otros."
    expected = target.removesuffix("and otros.") + "y otros."
    assert builder._contextualize(target, "es", source) == expected
    assert builder._contextualize(expected, "es", source) == expected


def test_citation_nested_inside_rst_code_restores_every_literal(monkeypatch):
    import build_i18n_catalogs as builder
    monkeypatch.setattr(builder, "_reviewed_translation", lambda *_: None)
    notice = "``(Copyright 2023 A and B)``"
    title = "``'accurate and efficient spot detection in microscopy'``"
    source = f"Use {notice} and {title}, doi:10.1038/s41592-025-02662-x."
    target = f"Utilice {notice} y {title}, doi:10.1038/s41592-025-02662-x."
    assert builder._contextualize(target, "es", source) == target


@pytest.mark.parametrize("language,source,screen_target,approved_target", [
    ('fr', 'TIFF, named like E_01_f9_c1_x405_mBlue_z1_t1.tif. Naming: incell_full_form. Fills the Import screen: images, cell / nucleus / pathogen masks and a measurements table, then previews.\n\nSynthetic pixels and constructed names demonstrate the documented Full Form convention; these are not acquired microscope images.', 'TIFF, avec des noms comme E_01_f9_c1_x405_mBlue_z1_t1.tif. Convention de nommage : incell_full_form. Remplit l’écran d’importation avec les images, les masques de cellule / noyau / pathogène et une table de mesures, puis affiche un aperçu.\n\nLes pixels synthétiques et les noms construits illustrent la convention Full Form documentée ; il ne s’agit pas d’images acquises au microscope.', 'TIFF, avec des noms comme E_01_f9_c1_x405_mBlue_z1_t1.tif. Convention de nommage : incell_full_form. Remplit la fenêtre d’importation avec les images, les masques de cellule / noyau / pathogène et une table de mesures, puis affiche un aperçu.\n\nLes pixels synthétiques et les noms construits illustrent la convention Full Form documentée ; il ne s’agit pas d’images acquises au microscope.'),
    ('fr', 'TIFF, named like data/--W00002--P00009--Z00000--T00000--DAPI.tif. Naming: scanr_indexed. Fills the Import screen: images, cell / nucleus / pathogen masks and a measurements table, then previews.\n\nSynthetic pixels use an explicit plate/index/well CSV: W00002 is E01 and W00007 is E02. The indices do not imply plate geometry.', 'TIFF, avec des noms comme data/--W00002--P00009--Z00000--T00000--DAPI.tif. Convention de nommage : scanr_indexed. Remplit l’écran d’importation avec les images, les masques de cellule / noyau / pathogène et une table de mesures, puis affiche un aperçu.\n\nLes pixels synthétiques utilisent un CSV explicite avec plate/index/well : W00002 correspond à E01 et W00007 à E02. Les indices n’impliquent aucune géométrie de plaque.', 'TIFF, avec des noms comme data/--W00002--P00009--Z00000--T00000--DAPI.tif. Convention de nommage : scanr_indexed. Remplit la fenêtre d’importation avec les images, les masques de cellule / noyau / pathogène et une table de mesures, puis affiche un aperçu.\n\nLes pixels synthétiques utilisent un CSV explicite avec plate/index/well : W00002 correspond à E01 et W00007 à E02. Les indices n’impliquent aucune géométrie de plaque.'),
    ('hi', 'TIFF, named like E_01_f9_c1_x405_mBlue_z1_t1.tif. Naming: incell_full_form. Fills the Import screen: images, cell / nucleus / pathogen masks and a measurements table, then previews.\n\nSynthetic pixels and constructed names demonstrate the documented Full Form convention; these are not acquired microscope images.', 'TIFF, जिसका नाम E_01_f9_c1_x405_mBlue_z1_t1.tif जैसा है। नामकरण: incell_full_form। आयात स्क्रीन में छवियाँ, सेल / न्यूक्लियस / पैथोजन मास्क और मापन तालिका भरता है, फिर पूर्वावलोकन दिखाता है।\n\nकृत्रिम पिक्सेल और बनाए गए नाम प्रलेखित Full Form नामकरण परंपरा का उदाहरण देते हैं; ये माइक्रोस्कोप से ली गई छवियाँ नहीं हैं।', 'TIFF, जिसका नाम E_01_f9_c1_x405_mBlue_z1_t1.tif जैसा है। नामकरण: incell_full_form। आयात विंडो में छवियाँ, सेल / न्यूक्लियस / पैथोजन मास्क और मापन तालिका भरता है, फिर पूर्वावलोकन दिखाता है।\n\nकृत्रिम पिक्सेल और बनाए गए नाम प्रलेखित Full Form नामकरण परंपरा का उदाहरण देते हैं; ये माइक्रोस्कोप से ली गई छवियाँ नहीं हैं।'),
    ('hi', 'TIFF, named like data/--W00002--P00009--Z00000--T00000--DAPI.tif. Naming: scanr_indexed. Fills the Import screen: images, cell / nucleus / pathogen masks and a measurements table, then previews.\n\nSynthetic pixels use an explicit plate/index/well CSV: W00002 is E01 and W00007 is E02. The indices do not imply plate geometry.', 'TIFF, जिसका नाम data/--W00002--P00009--Z00000--T00000--DAPI.tif जैसा है। नामकरण: scanr_indexed। आयात स्क्रीन में छवियाँ, सेल / न्यूक्लियस / पैथोजन मास्क और मापन तालिका भरता है, फिर पूर्वावलोकन दिखाता है।\n\nकृत्रिम पिक्सेल एक स्पष्ट plate/index/well CSV का उपयोग करते हैं: W00002 का वेल E01 है और W00007 का वेल E02 है। सूचकांक प्लेट की ज्यामिति नहीं दर्शाते।', 'TIFF, जिसका नाम data/--W00002--P00009--Z00000--T00000--DAPI.tif जैसा है। नामकरण: scanr_indexed। आयात विंडो में छवियाँ, सेल / न्यूक्लियस / पैथोजन मास्क और मापन तालिका भरता है, फिर पूर्वावलोकन दिखाता है।\n\nकृत्रिम पिक्सेल एक स्पष्ट plate/index/well CSV का उपयोग करते हैं: W00002 का वेल E01 है और W00007 का वेल E02 है। सूचकांक प्लेट की ज्यामिति नहीं दर्शाते।'),
    ('es', 'TIFF, named like E_01_f9_c1_x405_mBlue_z1_t1.tif. Naming: incell_full_form. Fills the Import screen: images, cell / nucleus / pathogen masks and a measurements table, then previews.\n\nSynthetic pixels and constructed names demonstrate the documented Full Form convention; these are not acquired microscope images.', 'TIFF, con nombres como E_01_f9_c1_x405_mBlue_z1_t1.tif. Convención de nombres: incell_full_form. Rellena la pantalla de importación con imágenes, máscaras de célula / núcleo / patógeno y una tabla de mediciones; después muestra una vista previa.\n\nLos píxeles sintéticos y los nombres construidos demuestran la convención Full Form documentada; no son imágenes adquiridas con un microscopio.', 'TIFF, con nombres como E_01_f9_c1_x405_mBlue_z1_t1.tif. Convención de nombres: incell_full_form. Rellena la ventana de importación con imágenes, máscaras de célula / núcleo / patógeno y una tabla de mediciones; después muestra una vista previa.\n\nLos píxeles sintéticos y los nombres construidos demuestran la convención Full Form documentada; no son imágenes adquiridas con un microscopio.'),
    ('es', 'TIFF, named like data/--W00002--P00009--Z00000--T00000--DAPI.tif. Naming: scanr_indexed. Fills the Import screen: images, cell / nucleus / pathogen masks and a measurements table, then previews.\n\nSynthetic pixels use an explicit plate/index/well CSV: W00002 is E01 and W00007 is E02. The indices do not imply plate geometry.', 'TIFF, con nombres como data/--W00002--P00009--Z00000--T00000--DAPI.tif. Convención de nombres: scanr_indexed. Rellena la pantalla de importación con imágenes, máscaras de célula / núcleo / patógeno y una tabla de mediciones; después muestra una vista previa.\n\nLos píxeles sintéticos utilizan un CSV explícito con plate/index/well: W00002 es E01 y W00007 es E02. Los índices no implican la geometría de la placa.', 'TIFF, con nombres como data/--W00002--P00009--Z00000--T00000--DAPI.tif. Convención de nombres: scanr_indexed. Rellena la ventana de importación con imágenes, máscaras de célula / núcleo / patógeno y una tabla de mediciones; después muestra una vista previa.\n\nLos píxeles sintéticos utilizan un CSV explícito con plate/index/well: W00002 es E01 y W00007 es E02. Los índices no implican la geometría de la placa.'),
], ids=["fr-full-form", "fr-indexed-scanr", "hi-full-form", "hi-indexed-scanr",
        "es-full-form", "es-indexed-scanr"])
def test_import_screen_context_preserves_gui_meaning_and_protected_inputs(
    language, source, screen_target, approved_target, monkeypatch,
):
    import build_i18n_catalogs as builder

    monkeypatch.setattr(builder, "_reviewed_translation", lambda *_: None)
    assert builder._gui_screen_source(source)
    for target in (screen_target, approved_target):
        assert builder._contextualize(target, language, source) == target
        assert builder._contextualize(
            builder._contextualize(target, language, source), language, source,
        ) == target
        assert builder._syntax_preserved(source, target)
        assert not builder._semantic_false_friends(source, target, language)
        assert not builder._syntax_preserved(source, target.replace(".tif", ".png"))
        if "plate/index/well" in source:
            csv_source = "CSV columns: ``plate,index,well``."
            assert builder._syntax_preserved(csv_source, csv_source)
            assert not builder._syntax_preserved(
                csv_source, csv_source.replace("plate,index,well", "plate,row,well"),
            )


@pytest.mark.parametrize("language,ui_target,assay_target", [
    ("fr", "L’écran Qt affiche les paramètres.", "Le criblage Qt affiche les paramètres."),
    ("hi", "Qt स्क्रीन सेटिंग दिखाती है।", "Qt स्क्रीनिंग सेटिंग दिखाती है।"),
    ("es", "La pantalla Qt muestra ajustes.", "El cribado Qt muestra ajustes."),
])
def test_ui_screen_preservation_keeps_scientific_correction_and_meaning_gate(
    language, ui_target, assay_target, monkeypatch,
):
    import build_i18n_catalogs as builder

    monkeypatch.setattr(builder, "_reviewed_translation", lambda *_: None)
    source = "The Qt screen shows settings."
    assert builder._contextualize(ui_target, language, source) == ui_target
    if language != "es":
        assert "gui-screen-as-scientific-screen" in builder._semantic_false_friends(
            source, assay_target, language,
        )
    scientific = "A pooled CRISPR screen reports hits."
    assert not builder._gui_screen_source(scientific)
    expected = {"fr": "criblage", "hi": "स्क्रीनिंग", "es": "cribado"}[language]
    noun = {"fr": "écran", "hi": "स्क्रीन", "es": "pantalla"}[language]
    assert builder._contextualize(noun, language, scientific) == expected


def test_mixed_gui_and_assay_screen_stays_outside_single_sense_classifier():
    import build_i18n_catalogs as builder

    source = "The GUI screen compares a pooled CRISPR screen."
    target = "L’écran GUI compare un criblage CRISPR."
    assert not builder._gui_screen_source(source)
    assert not builder._semantic_false_friends(source, target, "fr")
