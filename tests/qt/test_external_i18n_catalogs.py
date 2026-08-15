"""Coverage and freshness contracts for external localization catalogs."""
from __future__ import annotations

from importlib import import_module
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
LANGUAGES = ("sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr")
API_EXACT_TEXT_ALLOWLIST = {
    "spacr.align.CanvasSpec.shape",
    "spacr.errors.RunLedger.status",
    "spacr.hits.HitList.flag_counts",
    "spacr.macro.MacroStep.entry",
    "spacr.qt.iconset.themed_pixmap",
    "spacr.qt.screens.report.ReportScreen.output_format",
    "spacr.qt.settings_search.SettingsSearchBar.level",
    "spacr.qt.widgets.dose_response.DoseResponseResult.status",
    "spacr.qt.widgets.formula.Unary",
    "spacr.qt.widgets.plate_layout.PlateDesign.shape",
    "spacr.resources.home.versions._generators.common.app_map",
    "spacr.run_compare.HitList.by_key",
    "spacr.runctx.RunContext.__str__",
    "spacr.runctx.SkipRecord.__str__",
    "spacr.schema.field_index",
    "spacr.seg_qc.Scorecard.verdict",
}


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
        expected_hash_keys = {
            (table, str(key)) for table, keys in expected.items() for key in keys
        }
        assert set(catalog.SOURCE_HASHES) == expected_hash_keys
        assert catalog.SOURCE_HASHES == english.SOURCE_HASHES


def test_runtime_source_inventory_is_complete_before_optional_module_imports():
    tools_dir = str(ROOT / "tools")
    sys.path.insert(0, tools_dir)
    try:
        builder = import_module("build_i18n_catalogs")
    finally:
        sys.path.remove(tools_dir)

    sources = builder.canonical_sources()
    assert "barcode_qc" in sources["setting_tooltips"]
    from spacr.qt.i18n_catalogs import en
    assert set(sources["setting_tooltips"]) == set(en.SETTING_TOOLTIPS)


def test_runtime_source_inventory_is_stable_after_runctx_import():
    """Run controls are canonical regardless of prior module import order."""
    import spacr.runctx  # noqa: F401 - the import is the condition under test

    tools_dir = str(ROOT / "tools")
    sys.path.insert(0, tools_dir)
    try:
        builder = import_module("build_i18n_catalogs")
    finally:
        sys.path.remove(tools_dir)

    from spacr.qt.i18n_catalogs import en
    sources = builder.canonical_sources()
    assert set(sources["setting_tooltips"]) == set(en.SETTING_TOOLTIPS)
    assert "on_error" in sources["setting_tooltips"]


def test_runtime_rejects_a_localized_record_with_a_stale_source_hash(
    monkeypatch,
):
    from spacr.qt.i18n_catalogs import setting_tooltip
    from spacr.qt.i18n_catalogs import de as catalog
    from spacr.qt.i18n_catalogs.en import SETTING_TOOLTIPS

    key = "cell_diameter"
    source = SETTING_TOOLTIPS[key]
    assert setting_tooltip(key, source, "de")
    monkeypatch.setitem(
        catalog.SOURCE_HASHES, ("SETTING_TOOLTIPS", key), "stale"
    )
    assert setting_tooltip(key, source, "de") is None


def test_runtime_tooltips_have_no_exact_english_prose_fallbacks():
    english = import_module("spacr.qt.i18n_catalogs.en")
    for language in LANGUAGES:
        catalog = import_module(f"spacr.qt.i18n_catalogs.{language}")
        unchanged = [
            key for key, source in english.SETTING_TOOLTIPS.items()
            if catalog.SETTING_TOOLTIPS[key] == source
        ]
        assert not unchanged, f"{language}: {unchanged[:10]}"


def test_runtime_uses_external_static_and_context_keyed_setting_text():
    from spacr.qt.i18n import tr
    from spacr.qt.screens.settings_model import _translated_setting_name
    from spacr.qt.i18n_catalogs import setting_tooltip
    from spacr.qt.i18n_catalogs.en import SETTING_TOOLTIPS

    assert tr("Remove selected", "sv") == "Ta bort markerade"
    assert _translated_setting_name("plate", "zh_CN") == "孔板"
    assert _translated_setting_name("organelle_CP_prob", "ko") == "소기관 1 — CP"
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

    annotate_source = (
        "Open the Annotate screen first — it is what shows crops."
    )
    annotate = french_catalog.UI[annotate_source]
    assert "écran" in annotate and "vignettes" in annotate
    assert not any(word in annotate.casefold()
                   for word in ("criblage", "récolte", "culture"))

    run_source = "Every verdict here was written by the run that produced it -- "
    run_text = next(
        value for source, value in french_catalog.UI.items()
        if source.startswith(run_source)
    )
    assert "exécution" in run_text and "piste" not in run_text.casefold()

    mixed_source = next(
        source for source in french_catalog.UI
        if source.startswith("The simulator parameters this screen")
    )
    mixed_text = french_catalog.UI[mixed_source]
    assert "écran" in mixed_text and "criblage réel" in mixed_text


def test_runtime_catalogs_resolve_all_reviewed_false_friend_variants():
    tools_dir = str(ROOT / "tools")
    sys.path.insert(0, tools_dir)
    try:
        builder = import_module("build_i18n_catalogs")
    finally:
        sys.path.remove(tools_dir)

    sources = builder.canonical_sources()
    source_tables = {
        "SETTING_LABELS": sources["setting_labels"],
        "SETTING_TOOLTIPS": sources["setting_tooltips"],
        "CATEGORY_HELP": {source: source for source in sources["categories"]},
        "UI": {source: source for source in sources["ui"]},
        "MODULE_SUMMARIES": sources["module_summaries"],
    }
    for language in LANGUAGES:
        catalog = import_module(f"spacr.qt.i18n_catalogs.{language}")
        unresolved = []
        for table_name, table_sources in source_tables.items():
            table = getattr(catalog, table_name)
            for key, source in table_sources.items():
                value = table[key]
                if builder._contextualize(value, language, source) != value:
                    unresolved.append(f"{table_name}/{key}")
        assert not unresolved, f"{language}: {unresolved[:10]}"


def test_chinese_and_scientific_runtime_terms_are_contextual():
    from spacr.qt.i18n_catalogs import de, en, es, fr, zh_CN

    for key, source in en.SETTING_LABELS.items():
        value = zh_CN.SETTING_LABELS[key]
        if re.search(r"\bmasks?\b", source, re.IGNORECASE):
            assert "面具" not in value and "口罩" not in value
        if re.search(r"\bcells?\b", source, re.IGNORECASE):
            assert "电池" not in value
        if re.search(r"\bplates?\b", source, re.IGNORECASE):
            assert "板块" not in value
    for table_name, sources in (
        ("SETTING_TOOLTIPS", en.SETTING_TOOLTIPS),
        ("UI", {source: source for source in en.UI_SOURCES}),
    ):
        table = getattr(zh_CN, table_name)
        for key, source in sources.items():
            value = table[key]
            if re.search(r"\bmasks?\b", source, re.IGNORECASE):
                assert "面具" not in value and "口罩" not in value
            if re.search(r"\bcells?\b", source, re.IGNORECASE):
                assert "电池" not in value
            if re.search(r"\bplates?\b", source, re.IGNORECASE):
                assert "板块" not in value
            if re.search(r"\bguides?\b", source, re.IGNORECASE):
                assert "指南" not in value and "向导 RNA" not in value

    assert "golpe" not in es.SETTING_LABELS["power_hit_rate"].casefold()
    assert "Rennen" not in de.SETTING_TOOLTIPS["intermedeate_save"]
    assert "Formtor" not in de.SETTING_TOOLTIPS["organelle_ring_min_prominence"]
    question = "Ask a question about the table you are gating without leaving the screen."
    resolution = next(
        source for source in en.UI_SOURCES
        if "lower DPI for the screen" in source
    )
    assert "pantalla" in es.UI[question].casefold()
    assert "écran" in fr.UI[question].casefold()
    assert "pantalla" in es.UI[resolution].casefold()
    assert "écran" in fr.UI[resolution].casefold()


def test_api_doc_catalog_is_symbol_keyed_and_source_hashed():
    manifest = json.loads((
        ROOT / "docs" / "source" / "_static" / "i18n" / "api" / "en.json"
    ).read_text(encoding="utf-8"))
    assert manifest["schema"] == 2
    assert len(manifest["symbols"]) >= 6000
    for key, record in manifest["symbols"].items():
        assert key.startswith("spacr")
        assert re.fullmatch(r"[0-9a-f]{64}", record["source_sha256"])
        assert record["source_sha256"] == hashlib.sha256(
            record["text"].encode("utf-8")
        ).hexdigest()
        assert all(
            re.fullmatch(r"[0-9a-f]{64}", value)
            for value in record["source_blocks_sha256"]
        )
        assert record["text"].strip()
    for language in LANGUAGES:
        translated = json.loads((
            ROOT / "docs" / "source" / "_static" / "i18n" / "api"
            / f"{language}.json"
        ).read_text(encoding="utf-8"))
        assert set(translated["symbols"]) == set(manifest["symbols"])
        assert translated["schema"] == 2
        for key, record in translated["symbols"].items():
            assert record["source_sha256"] == manifest["symbols"][key]["source_sha256"]
            assert record["source_blocks_sha256"] == manifest["symbols"][key]["source_blocks_sha256"]
            assert record["text"].strip()
            if record["text"] == manifest["symbols"][key]["text"]:
                assert key in API_EXACT_TEXT_ALLOWLIST
        for key in (
            "spacr.spacrops.align_image_to_stitch",
            "spacr.utils.dense_mask_channel_positions",
        ):
            assert translated["symbols"][key]["text"] != manifest["symbols"][key]["text"]


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
