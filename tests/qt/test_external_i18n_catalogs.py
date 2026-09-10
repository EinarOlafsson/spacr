"""Coverage and freshness contracts for external localization catalogs."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from importlib import import_module
from pathlib import Path
from string import Formatter

ROOT = Path(__file__).resolve().parents[2]
LANGUAGES = ("sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr")
API_EXACT_TEXT_ALLOWLIST = {
    "spacr.align.CanvasSpec.shape",
    "spacr.errors.RunLedger.status",
    "spacr.gene_facts.Segment.text",
    "spacr.hits.HitList.flag_counts",
    "spacr.macro.MacroStep.entry",
    "spacr.qt.iconset.themed_pixmap",
    "spacr.qt.screens.report.ReportScreen.output_format",
    "spacr.qt.settings_search.SettingsSearchBar.level",
    "spacr.qt.widgets.dose_response.DoseResponseResult.status",
    "spacr.qt.widgets.formula.Unary",
    # "``'glow'``, ``'rainbow'`` or ``'beat'``" -- three literals, one
    # conjunction, and nothing else to translate. Admitted when the API
    # catalogs were regenerated on 2026-08-23; the docstring is older
    # than that and was simply never in a rebuilt catalog before.
    "spacr.qt.widgets.setup_card.SetupCard.mode",
    "spacr.qt.widgets.plate_layout.PlateDesign.shape",
    "spacr.run_compare.HitList.by_key",
    "spacr.runctx.RunContext.__str__",
    "spacr.runctx.SkipRecord.__str__",
    "spacr.seg_qc.Scorecard.verdict",
    "spacr.updater.PackageChange.describe",
}


def _format_fields(text: str) -> set[str]:
    return {
        name
        for _literal, name, _spec, _conversion in Formatter().parse(str(text))
        if name is not None
    }


def test_all_external_catalogs_preserve_runtime_placeholders():
    """A locale may not rename or drop a field that callers interpolate."""
    english = import_module("spacr.qt.i18n_catalogs.en")
    runtime_sources = {
        "SETTING_LABELS": english.SETTING_LABELS,
        "SETTING_TOOLTIPS": english.SETTING_TOOLTIPS,
        "CATEGORY_HELP": {
            source: source for source in english.CATEGORY_SOURCES
        },
        "UI": {source: source for source in english.UI_SOURCES},
        "MODULE_SUMMARIES": english.MODULE_SUMMARIES,
    }
    installer_sources = json.loads(
        (ROOT / "packaging" / "i18n" / "en.json").read_text(encoding="utf-8")
    )
    percent_field = re.compile(r"%(?:\d+\$)?[sd]")

    for language in LANGUAGES:
        catalog = import_module(f"spacr.qt.i18n_catalogs.{language}")
        for table_name, sources in runtime_sources.items():
            translated = getattr(catalog, table_name)
            for key, source in sources.items():
                assert _format_fields(translated[key]) == _format_fields(source), (
                    f"{language}/{table_name}/{key}: format fields changed"
                )
        installer = json.loads(
            (ROOT / "packaging" / "i18n" / f"{language}.json").read_text(
                encoding="utf-8"
            )
        )
        assert set(installer) == set(installer_sources)
        for key, source in installer_sources.items():
            assert percent_field.findall(installer[key]) == percent_field.findall(
                source
            ), f"{language}/installer/{key}: placeholders changed"


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
            assert all(
                str(value).strip()
                for value in getattr(catalog, table).values()
            )
        expected_hash_keys = {
            (table, str(key)) for table, keys in expected.items() for key in keys
        }
        assert set(catalog.SOURCE_HASHES) == expected_hash_keys
        assert catalog.SOURCE_HASHES == english.SOURCE_HASHES


def test_standalone_technical_identity_values_remain_exact_in_every_language():
    """Short configuration tokens may never be semanticized as prose."""

    expected_examples = {
        "cividis", "coolwarm", "inferno", "magma", "plasma", "turbo",
        "viridis", "otsu", "cellpose", "pymc", "numpyro", "umap", "tsne",
        "btrack", "trackastra", "trackpy", "ultrack", "slurm", "ssh",
        "torch", "png", "fdr_bh", "measurements.db", "seg_qc",
        "Amsgrad", "CSV", "Huber t", "Tensorboard",
        "gRNA", "gRNA CSV", "--partition=gpu --gres=gpu:1 --time=12:00:00",
        "2D / 3D UMAP", "EAF1_g1, EAF1_g2", "xD",
    }
    tools_dir = str(ROOT / "tools")
    sys.path.insert(0, tools_dir)
    try:
        identities = set(import_module("build_i18n_catalogs")._IDENTITY_TEXT)
    finally:
        sys.path.remove(tools_dir)
    assert expected_examples <= identities

    from spacr.qt.i18n import _ROWS, tr

    english = import_module("spacr.qt.i18n_catalogs.en")
    source_tables = {
        "SETTING_LABELS": english.SETTING_LABELS,
        "SETTING_TOOLTIPS": english.SETTING_TOOLTIPS,
        "CATEGORY_HELP": {value: value for value in english.CATEGORY_SOURCES},
        "UI": {value: value for value in english.UI_SOURCES},
        "MODULE_SUMMARIES": english.MODULE_SUMMARIES,
    }
    located = set()
    for language in LANGUAGES:
        catalog = import_module(f"spacr.qt.i18n_catalogs.{language}")
        for table_name, sources in source_tables.items():
            translated = getattr(catalog, table_name)
            for key, source in sources.items():
                if source not in identities:
                    continue
                located.add(source)
                assert translated[key] == source, (
                    f"{language}/{table_name}/{key}: technical identity "
                    f"{source!r} changed to {translated[key]!r}"
                )
        for source in identities & set(_ROWS):
            assert tr(source, language) == source
    assert located | (identities & set(_ROWS)) == identities


def test_runtime_catalogs_reject_known_cross_domain_contamination_markers():
    """Parliamentary/KDE corpus fragments must not re-enter UI catalogs."""

    markers = (
        "ordförande", "herr präsident", "@info: whatsthis",
        "description in lists", "en anglais seulement",
    )
    for language in LANGUAGES:
        catalog = import_module(f"spacr.qt.i18n_catalogs.{language}")
        values = (
            *catalog.SETTING_LABELS.values(),
            *catalog.SETTING_TOOLTIPS.values(),
            *catalog.CATEGORY_HELP.values(),
            *catalog.UI.values(),
            *catalog.MODULE_SUMMARIES.values(),
        )
        contaminated = [
            value for value in values
            if any(marker in str(value).casefold() for marker in markers)
        ]
        assert not contaminated, f"{language}: contaminated rows {contaminated[:3]}"


def test_reviewed_ui_rows_are_exact_in_regenerated_runtime_catalogs():
    """Reviewed captions and launch prose must override generated cache text."""

    tools_dir = str(ROOT / "tools")
    sys.path.insert(0, tools_dir)
    try:
        reviewed = import_module("i18n_reviewed_ui").REVIEWED_UI_TRANSLATIONS
    finally:
        sys.path.remove(tools_dir)

    english = import_module("spacr.qt.i18n_catalogs.en")
    from spacr.qt.i18n import _ROWS, VALID_LANGUAGE_CODES

    for language in LANGUAGES:
        catalog = import_module(f"spacr.qt.i18n_catalogs.{language}")
        located = set()
        for key, source in english.SETTING_LABELS.items():
            if source in reviewed:
                located.add(source)
                assert catalog.SETTING_LABELS[key] == reviewed[source][language]
        for source in english.UI_SOURCES:
            if source in reviewed:
                located.add(source)
                assert catalog.UI[source] == reviewed[source][language]
        for key, source in english.MODULE_SUMMARIES.items():
            if source in reviewed:
                located.add(source)
                assert catalog.MODULE_SUMMARIES[key] == reviewed[source][language]
        language_index = VALID_LANGUAGE_CODES.index(language) - 1
        for source, values in _ROWS.items():
            if source in reviewed:
                located.add(source)
                assert values[language_index] == reviewed[source][language]
        assert located == set(reviewed), (
            f"{language}: reviewed UI rows missing from regenerated tables: "
            f"{sorted(set(reviewed) - located)}"
        )


def test_runtime_source_inventory_is_complete_before_optional_module_imports():
    tools_dir = str(ROOT / "tools")
    sys.path.insert(0, tools_dir)
    try:
        builder = import_module("build_i18n_catalogs")
    finally:
        sys.path.remove(tools_dir)

    sources = builder.canonical_sources()
    assert "barcode_qc" in sources["setting_tooltips"]
    short_surface = builder._short_runtime_caption_sources(sources)
    assert set(sources["setting_labels"].values()) <= short_surface
    assert {
        "Joining predictions to measured objects and fitting held-out surrogate…",
        "Load table…",
        "Well",
        "Score the masks now",
    } <= short_surface
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


def test_dynamic_text_templates_enter_the_runtime_source_inventory():
    """Explicitly translatable status text must not bypass catalog building."""
    tools_dir = str(ROOT / "tools")
    sys.path.insert(0, tools_dir)
    try:
        builder = import_module("build_i18n_catalogs")
    finally:
        sys.path.remove(tools_dir)

    sources = set(builder.extract_static_ui_sources())
    assert any(
        source.startswith("Kernel containment is active for each trial:")
        for source in sources
    )
    assert any(
        source.startswith("Kernel containment is unavailable because")
        for source in sources
    )
    assert any(
        source.startswith("Available memory could not be measured")
        for source in sources
    )
    assert any(
        source.startswith("{available:.0f} GiB available")
        for source in sources
    )
    assert "Settings recipes…" in sources
    assert "Feature Dictionary…" in sources


def test_structured_gene_tile_text_enters_the_runtime_source_inventory():
    tools_dir = str(ROOT / "tools")
    sys.path.insert(0, tools_dir)
    try:
        builder = import_module("build_i18n_catalogs")
    finally:
        sys.path.remove(tools_dir)

    sources = set(builder.canonical_sources()["ui"])
    assert "gene id" in sources
    assert "effect (coefficient)" in sources
    assert any(source.startswith("guide {guide} is not in the gRNA reference")
               for source in sources)


def test_settings_model_explainers_enter_the_runtime_source_inventory():
    """Every component translated by the dynamic explainer is canonical."""
    tools_dir = str(ROOT / "tools")
    sys.path.insert(0, tools_dir)
    try:
        builder = import_module("build_i18n_catalogs")
    finally:
        sys.path.remove(tools_dir)

    from spacr.qt.screens import settings_model

    sources = set(builder.canonical_sources()["ui"])
    expected = set(settings_model._SETTINGS_MODEL_UI_SOURCES)
    assert len(expected) >= 80
    assert expected <= sources
    assert settings_model._MIXED_GUIDE_OUTPUT_NOTE in sources
    assert settings_model._MIXED_MULTIPLE_TESTING_NOTE in sources
    assert settings_model.INFORMATION_LIMIT_NOTE in sources


def test_every_set_translatable_text_call_has_static_catalog_sources():
    """Dynamic chrome may not hide an English template behind a variable."""
    tools_dir = str(ROOT / "tools")
    sys.path.insert(0, tools_dir)
    try:
        builder = import_module("build_i18n_catalogs")
    finally:
        sys.path.remove(tools_dir)

    from spacr.qt.i18n import _ROWS

    known = set(builder.extract_static_ui_sources()) | set(_ROWS)
    unresolved = []
    missing = []
    checked = 0
    for path in sorted((ROOT / "spacr" / "qt").rglob("*.py")):
        if "i18n_catalogs" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        constants = {}
        for statement in tree.body:
            if (isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)):
                constants[statement.targets[0].id] = statement.value
            elif (isinstance(statement, ast.AnnAssign)
                  and isinstance(statement.target, ast.Name)
                  and statement.value is not None):
                constants[statement.target.id] = statement.value
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if builder._call_name(node) != "set_translatable_text":
                continue
            checked += 1
            arguments = list(builder._candidate_arguments(
                node, "set_translatable_text"))
            values = [
                value.strip()
                for argument in arguments
                for value in builder._literal_strings(argument, constants)
                if builder._looks_translatable(value)
            ]
            location = f"{path.relative_to(ROOT)}:{node.lineno}"
            if not values:
                unresolved.append(location)
            missing.extend(
                f"{location}: {value!r}" for value in values
                if value not in known
            )
    assert checked >= 15
    assert not unresolved, "non-static templates: " + ", ".join(unresolved)
    assert not missing, "templates outside catalogs:\n" + "\n".join(missing)


def test_supported_runtime_languages_share_one_source_registry():
    """The selector, generator, and runtime loader expose the same locales."""
    tools_dir = str(ROOT / "tools")
    sys.path.insert(0, tools_dir)
    try:
        builder = import_module("build_i18n_catalogs")
    finally:
        sys.path.remove(tools_dir)

    from spacr.qt.i18n import VALID_LANGUAGE_CODES
    from spacr.qt.i18n_catalogs import CATALOG_LANGUAGES

    generated = tuple(builder.MODEL_SPECS)
    assert generated == CATALOG_LANGUAGES == VALID_LANGUAGE_CODES[1:]
    assert set(builder.NATIVE_LANGUAGE_NAMES) == set(generated)


def test_model_explainer_uses_exact_templates_and_preserves_values(
    monkeypatch,
):
    """The dynamic box translates whole records, never isolated terms."""
    from spacr.qt import i18n
    from spacr.qt.screens import settings_model

    exact = {
        "MODEL:": "MODELL:",
        settings_model._MODE_NOTES["mixed"]: "EXAKTE MODELLBESCHREIBUNG",
        settings_model._MIXED_GUIDE_OUTPUT_NOTE: "EXAKTE AUSGABEGRENZE",
        settings_model._MIXED_COST_NOTE_TEMPLATE: (
            "EXAKTE KOSTEN: {small_genes}/{small_wells}; "
            "{small_ratio:g}x; {guides}/{wells}"
        ),
    }
    monkeypatch.setattr(
        i18n, "_exact_translation",
        lambda source, language: exact.get(source) if language == "de" else None,
    )
    monkeypatch.setattr(
        i18n, "_term_translation",
        lambda _source, _language: "PARTIAL TRANSLATION MUST NOT APPEAR",
    )

    html = settings_model.regression_model_explainer_html(
        "mixed", language="de")
    assert "MODELL:" in html
    assert "EXAKTE MODELLBESCHREIBUNG" in html
    assert "EXAKTE AUSGABEGRENZE" in html
    assert "EXAKTE KOSTEN: 40/400; 54x; 823/610" in html
    assert settings_model.INFORMATION_LIMIT_NOTE in html
    assert "PARTIAL TRANSLATION MUST NOT APPEAR" not in html


def test_every_explainer_render_path_uses_its_declared_source(monkeypatch):
    """Every runtime branch must consume the inventoried source templates."""
    from spacr.qt.screens import settings_model

    seen = set()

    def record(source, language=None, **values):
        del language
        seen.add(str(source))
        try:
            return str(source).format(**values)
        except (IndexError, KeyError, ValueError):
            return str(source)

    monkeypatch.setattr(settings_model, "_translated_ui_text", record)
    for model in (*settings_model._MODE_NOTES, "not_a_model"):
        for level in settings_model.REGRESSION_LEVELS:
            settings_model.regression_model_explainer_html(model, level)
            settings_model.regression_model_explainer(model, level)
    settings_model.permutation_test_explainer_html()
    settings_model.permutation_test_explainer()

    assert seen == set(settings_model._SETTINGS_MODEL_UI_SOURCES)


def test_existing_explainer_requests_a_rerender_on_language_change(qapp):
    """The generic widget-tree pass must reach an already-open rich panel."""
    from spacr.qt.i18n import retranslate_widget_tree
    from spacr.qt.screens.app_screen import _ExplainerBrowser

    class Owner:
        def __init__(self):
            self.languages = []

        def refresh(self, language):
            self.languages.append(language)

    owner = Owner()
    box = _ExplainerBrowser(owner.refresh)
    retranslate_widget_tree(box, "de")
    assert owner.languages == ["de"]
    box.close()
    qapp.processEvents()


def test_model_explainer_keeps_language_on_internal_api_links():
    from spacr.qt.screens.settings_model import model_api_link

    _name, internal = model_api_link("group_lasso", "de")
    _name, external = model_api_link("mixed", "de")
    assert internal.endswith("/spacr/group_lasso/index.html?lang=de")
    assert "statsmodels.org" in external
    assert "?lang=" not in external


def test_runtime_rejects_a_localized_record_with_a_stale_source_hash(
    monkeypatch,
):
    from spacr.qt.i18n_catalogs import de as catalog
    from spacr.qt.i18n_catalogs import setting_tooltip
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


def test_runtime_catalogs_have_no_unreviewed_exact_english_fallbacks():
    """Every translatable runtime row resolves or is reviewed as identity."""
    tools_dir = str(ROOT / "tools")
    sys.path.insert(0, tools_dir)
    try:
        builder = import_module("build_i18n_catalogs")
    finally:
        sys.path.remove(tools_dir)

    english = import_module("spacr.qt.i18n_catalogs.en")
    source_tables = {
        "SETTING_LABELS": english.SETTING_LABELS,
        "SETTING_TOOLTIPS": english.SETTING_TOOLTIPS,
        "CATEGORY_HELP": {
            source: source for source in english.CATEGORY_SOURCES
        },
        "UI": {source: source for source in english.UI_SOURCES},
        "MODULE_SUMMARIES": english.MODULE_SUMMARIES,
    }
    for language in LANGUAGES:
        catalog = import_module(f"spacr.qt.i18n_catalogs.{language}")
        for table_name, sources in source_tables.items():
            table = getattr(catalog, table_name)
            unchanged = [
                key
                for key, source in sources.items()
                if (
                    table[key] == source
                    and builder._looks_translatable(source)
                    and builder._reviewed_translation(source, language)
                    != source
                )
            ]
            assert not unchanged, (
                f"{language}/{table_name}: {unchanged[:10]}"
            )


def test_runtime_catalogs_need_no_incremental_repairs():
    """The release catalogs are a fixed point of every current hard gate."""
    tools_dir = str(ROOT / "tools")
    sys.path.insert(0, tools_dir)
    try:
        builder = import_module("build_i18n_catalogs")
    finally:
        sys.path.remove(tools_dir)

    sources = builder.canonical_sources()
    for language in LANGUAGES:
        invalid = builder._invalid_catalog_sources(language, sources)
        assert not invalid, f"{language}: {sorted(invalid)[:10]}"


def test_runtime_uses_external_static_and_context_keyed_setting_text():
    from spacr.qt.i18n import tr
    from spacr.qt.i18n_catalogs import setting_tooltip
    from spacr.qt.i18n_catalogs.en import SETTING_TOOLTIPS
    from spacr.qt.screens.settings_model import _translated_setting_name

    assert tr("Remove selected", "sv") == "Ta bort markerade"
    assert _translated_setting_name("plate", "zh_CN") == "孔板"
    # THE REVIEWED TERM, not a bare acronym. `_REVIEWED_UI_TRANSLATIONS`
    # renders "Cp prob" as "<term> (CP)" in ALL NINE languages -- Swedish
    # "Cellsannolikhet (CP)", German "Zellwahrscheinlichkeit (CP)", Korean
    # "세포 확률(CP)" -- so the composed name carries the reviewed term and not
    # the English acronym alone. This line previously expected "소기관 1 — CP",
    # which was written while `_translated_setting_name` did not translate the
    # suffix AT ALL and every locale rendered "Organelle 1 — Cp prob"; the
    # expectation described the intended repair rather than the reviewed
    # vocabulary. The suffix is now translated for every language, so the
    # assertion names what the reviewed catalog actually says.
    assert (
        _translated_setting_name("organelle_CP_prob", "ko")
        == "소기관 1 — 세포 확률(CP)"
    )
    assert _translated_setting_name("FT", "sv") == "Flödeströskel (FT)"
    key = "cell_diameter"
    source = SETTING_TOOLTIPS[key]
    translated = setting_tooltip(key, source, "de")
    assert translated and translated != source
    # A changed source cannot display a stale translation.
    assert setting_tooltip(key, source + " changed", "de") is None


def test_four_legacy_organelle_slots_stay_materialized_with_a_zero_default():
    """Catalog breadth is independent of how many slots a fresh form shows."""
    from spacr.organelle_types import (
        CATALOGUED_ORGANELLE_SLOTS,
        DEFAULT_NUMBER_OF_ORGANELLES,
    )
    from spacr.qt.i18n_catalogs import en, setting_label, setting_tooltip

    assert DEFAULT_NUMBER_OF_ORGANELLES == 0
    assert CATALOGUED_ORGANELLE_SLOTS == 4
    keys = (
        "organelle_channel",
        "organelle_diameter",
        "organelle_type",
        "organelleb_diameter",
    )
    for key in keys:
        assert key in en.SETTING_LABELS
        assert key in en.SETTING_TOOLTIPS
        for language in ("de", "fr"):
            label_source = en.SETTING_LABELS[key]
            tooltip_source = en.SETTING_TOOLTIPS[key]
            label = setting_label(key, label_source, language)
            tooltip = setting_tooltip(key, tooltip_source, language)
            assert label and label != label_source
            assert tooltip and tooltip != tooltip_source


def test_higher_organelle_slots_reuse_one_source_bound_translation():
    """Generated slots stay translated without expanding every catalog."""
    from spacr.object_roles import setting_label as english_setting_label
    from spacr.qt.i18n_catalogs import en, setting_label, setting_tooltip
    from spacr.settings import tooltips

    key = "organellee_channel"  # slot 5: first above the materialized four
    tooltip_key = "organellee_log_max_sigma"
    label_source = english_setting_label(key)
    tooltip_source = re.sub(
        r"^\s*\([^)]*\)\s*[-–:]?\s*", "", tooltips[tooltip_key]
    ).strip()

    assert key not in en.SETTING_LABELS
    assert tooltip_key not in en.SETTING_TOOLTIPS
    label = setting_label(key, label_source, "de")
    tooltip = setting_tooltip(tooltip_key, tooltip_source, "de")
    assert label == "Organelle 5 — Kanal"
    assert tooltip and tooltip != tooltip_source
    assert "organellee_" in tooltip

    # The alias is as source-bound as a materialized record: arbitrary label
    # or tooltip changes cannot receive the primary slot's old translation.
    assert setting_label(key, label_source + " changed", "de") is None
    assert setting_tooltip(
        tooltip_key, tooltip_source + " changed", "de"
    ) is None


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

    run_source = (
        "Summarizes quality-control results stored by completed runs; "
        "opening this screen does not recompute masks or measurements. "
        "An out-of-date result was produced from older inputs. A missing "
        "result indicates that the corresponding check has not been run "
        "and must not be interpreted as a passing result."
    )
    run_text = french_catalog.UI[run_source]
    assert "contrôle" in run_text and "qualité" in run_text
    assert "piste" not in run_text.casefold()

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

    power_labels = [
        es.SETTING_LABELS[key]
        for key, source in en.SETTING_LABELS.items()
        if "power" in source.casefold()
    ]
    assert power_labels
    assert all("golpe" not in value.casefold() for value in power_labels)
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
    """Every language is one click from README.rst, THROUGH the picker page.

    This used to require all nine ``docs/i18n/readme/README.<lang>.rst`` paths
    to appear in README.rst itself, and instruction 361 made that impossible
    on purpose. The maintainer asked for the links to be a dropdown; on GitHub
    a dropdown is ``<details>``/``<summary>``, and README.rst cannot carry it,
    because github/markup renders reStructuredText with raw HTML disabled --
    ``.. raw:: html`` is refused and a literal ``<details>`` is escaped to
    visible text. So the menu lives in ``docs/i18n/readme/README.md``, which
    is Markdown and does render it, and README.rst carries ONE link to it.

    Asserting the old shape here would have demanded exactly what
    ``tests/test_the_language_picker_is_a_dropdown.py`` forbids in 28 tests.
    The requirement did not disappear -- every language must still be
    reachable and every page must still be a real translation -- so it is
    checked where it now lives.
    """
    readme = (ROOT / "README.rst").read_text(encoding="utf-8")
    picker_relative = "docs/i18n/readme/README.md"
    assert picker_relative in readme, "README.rst does not link the picker"
    assert "docs/i18n/TRANSLATION_MODELS.md" in readme

    picker = (ROOT / picker_relative).read_text(encoding="utf-8")
    for language in LANGUAGES:
        assert f"README.{language}.rst" in picker, (
            f"{language} is not on the picker page, so it is unreachable")
        translated = ROOT / f"docs/i18n/readme/README.{language}.rst"
        assert translated.is_file() and translated.stat().st_size > 10_000


def test_api_language_selector_supports_all_catalog_languages():
    script = (
        ROOT / "docs" / "source" / "_static" / "api_i18n.js"
    ).read_text(encoding="utf-8")
    assert "new URLSearchParams(location.search).get(\"lang\")" in script
    for language in ("en",) + LANGUAGES:
        assert re.search(rf"\b{re.escape(language)}\s*:", script)
