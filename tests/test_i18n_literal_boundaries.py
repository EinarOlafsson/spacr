"""Quoted API literals remain visible beside target-language grammar."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from build_i18n_catalogs import _contextualize, _syntax_preserved  # noqa: E402


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
