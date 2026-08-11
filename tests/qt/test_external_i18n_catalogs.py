"""Coverage and freshness contracts for external localization catalogs."""
from __future__ import annotations

from importlib import import_module
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
LANGUAGES = ("sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr")


def test_external_runtime_catalogs_have_exact_current_source_keys():
    english = import_module("spacr.qt.i18n_catalogs.en")
    expected = {
        "SETTING_LABELS": set(english.SETTING_LABELS),
        "SETTING_TOOLTIPS": set(english.SETTING_TOOLTIPS),
        "CATEGORY_HELP": set(english.CATEGORY_SOURCES),
        "UI": set(english.UI_SOURCES),
        "MODULE_SUMMARIES": set(english.MODULE_SUMMARIES),
    }
    for language in LANGUAGES:
        catalog = import_module(f"spacr.qt.i18n_catalogs.{language}")
        for table, keys in expected.items():
            assert set(getattr(catalog, table)) == keys
            assert all(str(value).strip() for value in getattr(catalog, table).values())


def test_runtime_uses_external_static_and_context_keyed_setting_text():
    from spacr.qt.i18n import tr
    from spacr.qt.screens.settings_model import _translated_setting_name
    from spacr.qt.i18n_catalogs import setting_tooltip
    from spacr.qt.i18n_catalogs.en import SETTING_TOOLTIPS

    assert tr("Remove selected", "sv") == "Ta bort markerade"
    assert _translated_setting_name("plate", "zh_CN") == "孔板"
    assert _translated_setting_name("organelle_CP_prob", "ko") == "소기관 — CP"
    assert _translated_setting_name("FT", "sv") == "Flödeströskel (FT)"
    key = "cell_diameter"
    source = SETTING_TOOLTIPS[key]
    translated = setting_tooltip(key, source, "de")
    assert translated and translated != source
    # A changed source cannot display a stale translation.
    assert setting_tooltip(key, source + " changed", "de") is None


def test_visible_setting_labels_use_app_context(qapp):
    from PySide6.QtWidgets import QLabel
    from spacr.qt.i18n import retranslate_widget_tree
    from spacr.qt.i18n_catalogs import setting_label
    from spacr.qt.i18n_catalogs.en import SETTING_LABELS

    key = "cytoplasm"
    app_key = "measure"
    source = SETTING_LABELS[f"{app_key}.{key}"]
    expected = setting_label(key, source, "de", app_key)
    label = QLabel(source)
    label.setProperty("settingsAppKey", app_key)
    label.setProperty("settingKey", key)
    retranslate_widget_tree(label, "de")
    assert expected and label.text() == expected


def test_transient_dialogs_translate_when_shown(qapp, monkeypatch):
    from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout
    from spacr.qt.i18n import install_dialog_translation
    from spacr.qt.i18n_catalogs import ui_text

    source = "Choose folder for the demo dataset"
    expected = ui_text(source, "de")
    assert expected and expected != source
    monkeypatch.setenv("SPACR_LANGUAGE", "de")
    install_dialog_translation(qapp)
    dialog = QDialog()
    dialog.setWindowTitle(source)
    layout = QVBoxLayout(dialog)
    layout.addWidget(QLabel(source))
    dialog.show()
    qapp.processEvents()
    try:
        assert dialog.windowTitle() == expected
        assert dialog.findChild(QLabel).text() == expected
    finally:
        dialog.close()


def test_reviewed_scientific_terms_use_domain_context_not_false_friends():
    from spacr.qt.i18n import tr
    from spacr.qt.screens.curate import register as register_curate
    from spacr.qt.screens.gate_editor import (
        APP_NAME_TRANSLATIONS as gate_names,
    )
    from spacr.qt.screens.hit_list import APP_TRANSLATIONS as hit_names
    from spacr.qt.screens.power import APP_TRANSLATIONS as power_names

    assert tr("Segmentation", "hi") == "छवि विभाजन"
    assert tr("Cluster", "fr") == "Cluster"
    assert tr("Scan", "pt") == "Escanear"
    assert tr("Annotation", "ko") == "어노테이션"
    assert tr("Leakage audit", "fr") == "Audit des fuites de données"
    assert tr("Tracking", "is") == "Rakning"
    assert tr("Plate", "zh_CN") == "孔板"
    assert tr("Queue", "ko") == "대기열"
    assert tr("Viewer", "fr") == "Visionneuse"
    assert tr("Flow threshold", "zh_CN") == "流场阈值"
    assert tr("Minimum area", "de") == "Mindestfläche"
    assert tr("Save gates", "es") == "Guardar compuertas"
    assert tr("Recruitment", "zh_CN") == "募集分析"
    register_curate()
    assert tr("Curate", "es") == "Curación"
    assert tr("Curate", "pt") == "Curadoria"
    assert tr("Curate", "fr") == "Curation"
    assert hit_names[8] == "Liste des résultats"
    assert power_names[0].startswith("Statistisk")
    assert power_names[4].startswith("Potência")
    assert gate_names[7] == "Gate-ritill"
    assert gate_names[8] == "Éditeur de gates"

    from spacr.qt.i18n_catalogs import fr as french_catalog
    french_values = (
        *french_catalog.SETTING_LABELS.values(),
        *french_catalog.SETTING_TOOLTIPS.values(),
        *french_catalog.CATEGORY_HELP.values(),
        *french_catalog.UI.values(),
        *french_catalog.MODULE_SUMMARIES.values(),
    )
    assert not any("l'criblage" in value or "l’criblage" in value
                   for value in french_values)


def test_api_doc_catalog_is_symbol_keyed_and_source_hashed():
    manifest = json.loads((
        ROOT / "docs" / "source" / "_static" / "i18n" / "api" / "en.json"
    ).read_text(encoding="utf-8"))
    assert manifest["schema"] == 1
    assert len(manifest["symbols"]) >= 6000
    for key, record in manifest["symbols"].items():
        assert key.startswith("spacr")
        assert re.fullmatch(r"[0-9a-f]{64}", record["source_sha256"])
        assert record["text"].strip()
    for language in LANGUAGES:
        translated = json.loads((
            ROOT / "docs" / "source" / "_static" / "i18n" / "api"
            / f"{language}.json"
        ).read_text(encoding="utf-8"))
        assert set(translated["symbols"]) == set(manifest["symbols"])
        for key, record in translated["symbols"].items():
            assert record["source_sha256"] == manifest["symbols"][key]["source_sha256"]
            assert record["text"].strip()


def test_github_readme_links_every_external_language_page():
    readme = (ROOT / "README.rst").read_text(encoding="utf-8")
    for language in LANGUAGES:
        relative = f"docs/i18n/readme/README.{language}.rst"
        assert relative in readme
        translated = ROOT / relative
        assert translated.is_file() and translated.stat().st_size > 10_000
    assert "docs/i18n/TRANSLATION_MODELS.md" in readme


def test_api_language_selector_supports_all_catalog_languages():
    script = (
        ROOT / "docs" / "source" / "_static" / "api_i18n.js"
    ).read_text(encoding="utf-8")
    assert "new URLSearchParams(location.search).get(\"lang\")" in script
    for language in ("en",) + LANGUAGES:
        assert re.search(rf"\b{re.escape(language)}\s*:", script)
