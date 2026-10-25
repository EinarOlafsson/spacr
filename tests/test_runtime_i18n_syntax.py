"""Focused syntax contracts for generated runtime localization catalogs."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def test_physical_screen_pixel_budget_preserves_display_sense(monkeypatch):
    import build_i18n_catalogs as builder

    monkeypatch.setattr(builder, "_reviewed_translation", lambda *_: None)
    targets = {
        "es": "El renderizado se mantiene dentro del presupuesto de píxeles de la pantalla.",
        "pt": "A renderização permanece dentro do orçamento de píxeis da tela.",
        "fr": "Le rendu reste dans le budget de pixels de l’écran.",
        "zh_CN": "渲染工作保持在屏幕的像素预算内。",
        "hi": "रेंडरिंग स्क्रीन के पिक्सेल बजट के भीतर रहती है।",
    }
    for source in ("Rendering stays within the screen's pixel budget.",
                   "Rendering stays within the screen’s pixel budget."):
        assert builder._gui_screen_source(source)
        for language, target in targets.items():
            assert builder._contextualize(target, language, source) == target
            assert not builder._translation_rejection_reasons(source, target, language, force=True)
    assert not builder._gui_screen_source("A pooled CRISPR screen identifies genes.")
    assert builder._contextualize("A tela identifica genes.", "pt",
                                  "A pooled CRISPR screen identifies genes.") == "A triagem identifica genes."


def test_return_summary_about_a_genetic_screen_is_not_the_gui_screen():
    import build_i18n_catalogs as builder

    genetic = (
        "Return a screen with guide-to-well assignments permuted.",
        "Return the folder for one screen plate: ``<dest>/screen/plate<N>``.",
        "Return whether a plate identity collides within one screen.",
        "Return settings for re-running a screen with a new model.",
        "Return questions that cannot be answered from this screen's data.",
        "Return the canonical screen id, defaulting an absent one.",
    )
    for source in genetic:
        assert not builder._gui_screen_source(source), source
    for source in (
        "Leave ``queue`` for the next Make Masks screen that is built.",
        "Return to the home screen.",
        "Return the strip installed on ``screen``, or None.",
        "Rendering stays within the screen's pixel budget.",
    ):
        assert builder._gui_screen_source(source), source
    natural = {
        "ko": "가이드-웰 할당을 무작위로 섞은 스크리닝을 반환합니다.",
        "zh_CN": "返回将引导序列与孔的分配随机置换后的筛选。",
        "is": "Skilar skimun þar sem úthlutun leiðiraða á brunna er umröðuð.",
    }
    for language, target in natural.items():
        assert "ui-screen" not in builder._semantic_false_friends(
            genetic[0], target, language)


def test_retained_branded_theme_titles_preserve_names_without_allowing_prose(monkeypatch):
    import build_i18n_catalogs as builder

    monkeypatch.setattr(builder, "_reviewed_translation", lambda *_: None)
    for name in ("spaCR blobs", "spaCR aurora", "spaCR stratified", "spaCR spinn"):
        assert name in builder._IDENTITY_TEXT
        assert not builder._looks_translatable(name)
        for language in builder.MODEL_SPECS:
            assert not builder._translation_rejection_reasons(name, name, language)
    source = "The backdrop contains moving dots."
    assert "exact" in builder._translation_rejection_reasons(source, source, "sv", force=True)


def test_make_masks_tool_registry_and_fallback_names_enter_extraction(monkeypatch) -> None:
    from build_i18n_catalogs import _indirect_runtime_ui_sources
    from spacr.qt.screens import make_masks

    monkeypatch.setattr(make_masks, "TOOL_MODES", [
        *make_masks.TOOL_MODES,
        ("sentinel", "Registered tool sentinel", "brush"),
    ])
    monkeypatch.setattr(make_masks, "MODE_SENTINEL_FALLBACK", "fallback_sentinel", raising=False)
    sources = _indirect_runtime_ui_sources()
    assert {label for _mode, label, _icon in make_masks.tool_row_entries()} <= sources
    assert "Registered tool sentinel" in sources
    assert "Fallback sentinel" in sources


def test_volume_measurement_diagnostics_enter_runtime_catalog():
    from build_i18n_catalogs import extract_static_ui_sources

    sources = set(extract_static_ui_sources())
    assert {
        "3D measurement requires exactly three spatial axes.",
        "The channel voxel spacing does not match the source image: {path}.",
        "This image identity already belongs to other measurements; existing rows were preserved.",
        "Measuring 3D image set {current} of {total}...",
        "Measurements saved: {path}",
    } <= sources


def test_calibrated_batch_and_measurement_refusals_enter_runtime_catalog():
    import ast
    import build_i18n_catalogs as builder

    owners = {
        "core": {"preprocess_generate_masks"},
        "measure": {"_validate_measurement_coordinate_source",
                    "_calibrated_measurement_checkpoint",
                    "_publish_calibrated_measurement_directory",
                    "_calibrated_cell_statistics", "measure_calibrated_cells"},
        "pipeline_v2": {"_read_plane", "_calibrated_label_array",
                        "stream_originals_to_stack", "stream_masks_from_stack", "run_v2"},
        "psf_pipeline": {"_initialize_calibrated_output", "_validate_calibrated_batch_settings",
                         "_validate_unmix_plate_scope", "_plate_unmix_controls", "_plate_control_input",
                         "_prepare_segmentation_psf", "_SegmentationPSFSession",
                         "_cell_record_error", "_CalibratedCellDataset"},
    }
    expected = set()
    for module, names in owners.items():
        tree = ast.parse((ROOT / "spacr" / (module + ".py")).read_text(encoding="utf-8"))
        selected = [node for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        assert {node.name for node in selected} == names
        for owner in selected:
            expected.update(node.args[0].value for node in ast.walk(owner)
                            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                            and node.func.id == "tr" and node.args
                            and isinstance(node.args[0], ast.Constant)
                            and isinstance(node.args[0].value, str)
                            and builder._looks_translatable(node.args[0].value))
    assert expected
    assert expected <= set(builder.extract_static_ui_sources())


def test_make_masks_short_group_headings_enter_translation():
    import ast
    import build_i18n_catalogs as builder

    source = ROOT / "spacr/qt/screens/make_masks.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    captions = {node.args[0].value for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "tr" and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and builder._looks_translatable(node.args[0].value)}
    assert {"TOOLS", "IO"} <= captions
    for identifier in ("CPU", "GPU", "API", "XYZ", "RGBA", "N673"):
        assert not builder._looks_translatable(identifier)


def test_portuguese_well_locations_preserve_containers_and_repair_adverbs() -> None:
    """Container positions retain poço; adverbial well still repairs to bem."""
    from build_i18n_catalogs import _contextualize, _translation_rejection_reasons

    cases = (
        ("One well within a replicate, or one field within a well.",
         "Um poço dentro de uma réplica ou um campo dentro de um poço.",
         "Um poço dentro de uma réplica ou um campo dentro de um poço."),
        ("The well below the control well.",
         "O poço abaixo do poço de controle.",
         "O poço abaixo do poço de controle."),
        ("Values remain well within tolerance.",
         "Os valores permanecem poço dentro da tolerância.",
         "Os valores permanecem bem dentro da tolerância."),
    )
    for source, draft, expected in cases:
        assert _contextualize(draft, "pt", source) == expected
        assert _contextualize(expected, "pt", source) == expected
        assert not _translation_rejection_reasons(source, expected, "pt", force=True)


def _new_download_sources(language: str, reviewed: dict[str, str]) -> set[str]:
    """Account for the Import and synthetic Invasion review records separately."""
    sources: set[str] = set()
    for filename, expected in (("import-examples", 9), ("synthetic-invasion", 1)):
        document = json.loads((ROOT / "docs/i18n/reviewed/runtime" / language /
                               f"2026-09-21-{filename}.json").read_text(encoding="utf-8"))
        added = {record["source"] for record in document["records"]}
        # A record moved to retired_records (its English source changed) still
        # counts toward the historical totals, but is no longer live review.
        retired = {record["source"] for record in document.get("retired_records", [])
                   if str(record.get("retired", "")).startswith("2026-10-10")}
        assert len(document["records"]) + len(retired) == len(added | retired) == expected
        assert added <= reviewed.keys()
        assert not retired & reviewed.keys()
        assert not (added | retired) & sources
        sources.update(added | retired)
    # Item463 replaced the unsegmented-example tooltip. Preserve the original
    # two-record evidence and prove that only the superseded tooltip retired.
    archive = json.loads((ROOT / "features/data/463_retired_runtime_review_2026-09-27" /
                          f"{language}.json").read_text(encoding="utf-8"))["records"]
    old_invasion = {record["source"] for record in archive}
    assert len(archive) == len(old_invasion) == 2
    assert len(old_invasion - sources) == 1
    assert not (old_invasion - sources) & reviewed.keys()
    assert len(sources) == 10
    return sources


def _later_cohort_records(path, expected, language, reviewed):
    """Retain the frozen cohort bytes while proving two removed brush captions."""
    if path.name != "2026-09-22-lock-levels-parascape-controls.json":
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"]
        return json.loads(path.read_text(encoding="utf-8"))["records"], set()
    import gzip

    archive = ROOT / "tests/data/release_contracts/697_retired_runtime_controls_2026-10-15" / (language + ".json.gz")
    original_bytes = gzip.decompress(archive.read_bytes())
    assert hashlib.sha256(original_bytes).hexdigest() == expected["sha256"]
    document = json.loads(original_bytes)
    hashes = {
        "778abbd7904eeb2ed756fd39c45c0fb9bb631776ae43f029402d2546ed0a5574",
        "61ff206138ebad2ee38e9ec13a0b782892f5c2a81ff43c37d5739ef1472c4cf5",
    }
    retired = [row for row in document["records"] if row["source_sha256"] in hashes]
    assert len(retired) == len(hashes) == 2
    assert {row["source_sha256"] for row in retired} == hashes
    for row in retired:
        assert hashlib.sha256(row["source"].encode()).hexdigest() == row["source_sha256"]
    document["records"] = [row for row in document["records"] if row not in retired]
    expected_bytes = (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode()
    assert path.read_bytes() == expected_bytes
    sources = {row["source"] for row in retired}
    assert not sources & reviewed.keys()
    return document["records"], sources


def _later_cohort_retirements(language: str, reviewed: dict[str, str]) -> set[str]:
    report = json.loads((ROOT / "tests/data/release_contracts/411_runtime_review_cohorts_2026-09-23.json").read_text(encoding="utf-8"))["languages"][language]
    name = "2026-09-22-lock-levels-parascape-controls.json"
    expected = [row for row in report["files"] if row["name"] == name]
    assert len(expected) == 1
    path = ROOT / "docs/i18n/reviewed/runtime" / language / name
    records, retired = _later_cohort_records(path, expected[0], language, reviewed)
    assert len(records) + len(retired) == expected[0]["records"]
    return retired


def _superseded_enhancement_sources(language, reviewed):
    """Retain the one published display toggle replaced by explicit Apply."""
    document = json.loads((ROOT / "docs/i18n/reviewed/runtime" / language /
                           "2026-09-22-enhancement-and-scale.json").read_text(encoding="utf-8"))
    superseded = document["superseded_records"]
    assert len(superseded) == 1
    entry = superseded[0]
    assert set(entry) == {"reason", "record"}
    assert entry["reason"] == (
        "479: explicit Apply now gates enhancements for display and detection; "
        "Compare previews configured settings. Original reviewed text retained verbatim.")
    source = "Show the enhanced image"
    record = entry["record"]
    assert record == {
        "key": source,
        "source": source,
        "source_sha256": "c7e42861b48c38362b9cad0b778a53a42db6ab9330912bf469583ff32f199f5d",
        "table": "ui",
        "translation": {"sv": "Visa den förbättrade bilden",
                        "fr": "Afficher l’image améliorée"}[language],
    }
    assert hashlib.sha256(source.encode()).hexdigest() == record["source_sha256"]
    assert source not in reviewed
    return {source}


def _subsequent_review_sources(language: str, reviewed: dict[str, str]) -> set[str]:
    """Reconcile later accepted records without changing historical counts."""
    sources: set[str] = set()
    for filename, expected in (
        ("2026-09-21-channel-index-tooltip.json", 1),
        ("2026-09-21-annotate-opening.json", 2),
        ("2026-09-21-test-data-chooser.json", 3),
        ("2026-09-21-plaque-scale-conflict.json", 7),
        ("2026-09-22-view-gate-tooltip.json", 1),
        ("2026-09-22-download-status.json", 13),
        ("2026-09-22-shared-controls.json", 27),
        ("2026-09-22-workflow-pathways.json", 43),
        ("2026-09-22-figure-settings.json", 41),
        ("2026-09-22-home-sample-project.json", 2),
        ("2026-09-22-inversion-controls.json", 2 if language == "sv" else 1),
    ):
        document = json.loads((ROOT / "docs/i18n/reviewed/runtime" / language /
                               filename).read_text(encoding="utf-8"))
        added = {record["source"] for record in document["records"]}
        # A record moved to retired_records (its English source changed) still
        # counts toward the historical totals, but is no longer live review.
        retired = {record["source"] for record in document.get("retired_records", [])
                   if str(record.get("retired", "")).startswith("2026-10-10")}
        assert len(document["records"]) + len(retired) == len(added | retired) == expected
        assert added <= reviewed.keys()
        assert not retired & reviewed.keys()
        assert not (added | retired) & sources
        sources.update(added | retired)
    assert len(sources) == (142 if language == "sv" else 141)
    overlay = json.loads((ROOT / "docs/i18n/reviewed/runtime" / language /
                          "2026-09-22-plaque-overlays.json").read_text(encoding="utf-8"))
    overlay_sources = {record["source"] for record in overlay["records"]}
    # Plaque model has both a UI caption and a setting-label record.
    assert len(overlay["records"]) == 12
    assert len(overlay_sources) == 11
    assert overlay_sources <= reviewed.keys()
    assert not overlay_sources & sources
    sources.update(overlay_sources)
    assert len(sources) == (153 if language == "sv" else 152)
    panel = json.loads((ROOT / "docs/i18n/reviewed/runtime" / language /
                        "2026-09-22-detection-panel.json").read_text(encoding="utf-8"))
    panel_sources = {record["source"] for record in panel["records"]}
    panel_retired = _requested_detection_retirements(
        language, '2026-09-22-detection-panel.json')
    assert len(panel['records']) == len(panel_sources)
    assert len(panel_sources) + len(panel_retired) == 6
    assert panel_sources <= reviewed.keys()
    assert not panel_sources & sources
    sources.update(panel_sources)
    assert len(sources) + len(panel_retired) == (159 if language == "sv" else 158)
    # Instruction 316 retired the one sign-in-status record to _ROWS.
    for filename, expected in (("form-labels-a", 77), ("sign-in-status", 0),
                               ("enhancement-and-scale", 9),
                               ("organism-identities", 2),
                               ("threshold-and-histogram", 13)):
        document = json.loads((ROOT / "docs/i18n/reviewed/runtime" / language /
                               f"2026-09-22-{filename}.json").read_text(encoding="utf-8"))
        added = {record["source"] for record in document["records"]}
        superseded = (_superseded_enhancement_sources(language, reviewed)
                      if filename == "enhancement-and-scale" else set())
        assert len(document["records"]) == len(added)
        assert len(added | superseded) == len(added) + len(superseded) == expected
        assert added <= reviewed.keys()
        if filename == "form-labels-a":
            # The UI row reuses an already reviewed setting-label source.
            assert "Crop size" in added
            added.remove("Crop size")
        assert not (added | superseded) & sources
        sources.update(added | superseded)
    assert len(sources) + len(panel_retired) == (259 if language == "sv" else 258)
    report = json.loads((ROOT / "tests/data/release_contracts/411_runtime_review_cohorts_2026-09-23.json").read_text(encoding="utf-8"))["languages"][language]
    folder = ROOT / "docs/i18n/reviewed/runtime" / language
    later_sources: set[str] = set()
    later_retired: set[str] = set()
    later_names = {row["name"] for row in report["files"]}
    for row in report["files"]:
        path = folder / row["name"]
        records, retired = _later_cohort_records(path, row, language, reviewed)
        values = {record["source"] for record in records}
        assert len(records) + len(retired) == row["records"]
        assert len(values) + len(retired) == row["sources"]
        assert values <= reviewed.keys()
        later_sources.update(values)
        later_retired.update(retired)
    earlier_sources = {
        record["source"] for path in folder.glob("*.json") if path.name not in later_names
        for record in json.loads(path.read_text(encoding="utf-8"))["records"]
    }
    additions = later_sources - earlier_sources
    # 781 -> 780 on 2026-09-25: nightly merge 5c03e8bda removed one retired PSF
    # help record from 2026-09-23-psf-help.json (11 -> 10 records).
    # 780 -> 778 on 2026-10-01: the Make Masks re-layout rewrote two
    # FEATURES workflow-map phrases; their records were retired in place
    # (2026-09-23-workflow-map-phrases.json, 208 -> 206 records).
    # 778 -> 776 on 2026-10-02 (item 43): 603423d0a retired one more record
    # in place in each of 2026-09-23-pipeline-workflow-phrases.json (8 -> 7)
    # and 2026-09-23-workflow-map-phrases.json (206 -> 205); the receipt's
    # hashes and counts were rebound to those files.
    historical_additions = additions | later_retired
    assert not later_retired & earlier_sources
    assert len(historical_additions) == report["later_distinct_additions"] == 773  # 2026-10-24: -1, Timelapse moved under Mask's Time control, so its workflow-map guidance record retired; 2026-10-04: -2, 48c4133c8 withdrew the Plasmodium/Candida guide notes
    assert hashlib.sha256(json.dumps(sorted(historical_additions), ensure_ascii=False).encode()).hexdigest() == report["added_sources_sha256"]
    assert not sources & additions
    for filename, record_count, source_count in (
            # 3 -> 2 records on 2026-09-29: items 591-597 renamed the "Cloud"
            # category, so its caption record was deleted (key gone).
            ("2026-09-27-mask-cloud-category.json", 2, 1),
            ("2026-09-28-runtime-577-585.json", 20, 20)):
        document = json.loads((folder / filename).read_text(encoding="utf-8"))
        records = document["records"]
        values = {record["source"] for record in records}
        assert len(records) == record_count and len(values) == source_count
        assert values <= reviewed.keys()
        assert not values & (sources | additions)
        for record in records:
            assert record["source_sha256"] == hashlib.sha256(record["source"].encode()).hexdigest()
            assert reviewed[record["source"]] == record["translation"]
        sources.update(values)
    return sources | additions


def _compact_tooltip_sources(language: str) -> set[str]:
    """Two concise replacements retain the scientific review cohort's size."""
    path = ROOT / "docs/i18n/reviewed/runtime" / language / "2026-09-22-compact-tooltips.json"
    records = json.loads(path.read_text(encoding="utf-8"))["records"]
    assert len(records) == 2
    assert {record["table"] for record in records} == {"setting_tooltips"}
    assert {record["key"] for record in records} == {"annotation_source", "metadata_type"}
    return {record["source"] for record in records}


#: Sources retired by item 600b (the Features button's tooltip).
_RETIRED_BY_600B = {
    "Measure the masks you drew. Opens a table where each row is a field and "
    "each column is a channel or a mask type; the run goes through the "
    "Measure module itself, so the folders and the measurements database "
    "are the ones a Measure run produces.",
}



#: 2026-10-12: the seven narration captions of the deleted spacr-tutorial
#: still-frame renderer (spacr/qt/tutorial/scripts.py) left the inventory;
#: their reviewed records moved to retired_records with this dated reason.
TUTORIAL_RENDERER_RETIRED = (
    "2026-10-12: English caption removed with the spacr-tutorial still-frame "
    "renderer")

def _published_panel_retirements(language, filename):
    """Recognize only the exact preserved rows of the published panel removal."""
    hashes = (
        "eecb98b2dd4576250672cbceba3e1622303d8f87555486c985a3df474e8349fe",
        "a7a453fe3013e5f0d4d61aec261226d80aeac78e0ed6124f6514f51e65c1c3d4",
        "dd935b5d7fe48a39493b59e044ff62e36a0ca816576d970d263c3b7ea601d776",
        "6f5e747dba4dc3d4cc0a05b17936533f61b4bd23cedd56267396af17fe22d9f8",
        "bfd402b2f6f3812529b55596136d3a11c51616317e3b1cd999928e2d4eae7d3f",
        "ef897d0ef358f9b184a513da4ab465debe812d22ce608282b06b5bfe3e6d609f",
        "deec05063963965c40890d0b21b74cace845bb7fa0e561af9e3d1ebc65589d7f",
        "f7441f3a2c785aa2262308bdb4f92fa2c9106ceba32fb21639f2ccc895747bc0",
        "42958a5514188bdb1e8613d56d63acba59a710857f2735e7cb28781d5fac0906",
        "4bbf3ad8e30bbbc39c55d1b49ad3ebbd59692284ee35fd06abeecb76e4f7bcc3",
        "cfa215aae696efbba710060862248ae41321382ccc90709ea2beeb8fabd5cc0b",
        "a50a50796b80fc1e05551f491f493a6f95e6ab470343a88d0fabac5e9f2c42d4",
        "85d9fea65dffdf058adbdc703a373506ad275a50a9a8f025c270d557e41d20f2",
        "a8cc7870f3913195a9b2c8eac714a872b44fb7153faa34aebbfe1dd221e6f93e",
        "bb5fc33dbf7cd3f9ab2ba87cf48e7575a1a6c18bae01c505d0eac9a54028cfce",
    )
    folder = ROOT / "docs/i18n/reviewed/runtime" / language
    theme = folder / "archive/2026-10-06-home-cell-dino-save-themes"
    evidence = ROOT / "features/data"
    specifications = {
        ("sv", "2026-09-21-runtime-ui-refresh.json"): (
            theme / filename,
            "92aa0267ffa262e4adce2ba1e0f6612e13f73fa64d552807671b88797d1d569d", hashes),
        ("fr", "2026-09-21-runtime-second-slice.json"): (
            evidence / "450_451_retired_runtime_review_2026-09-27/fr" / filename,
            "f146191d22f5a420b28b25916f230d534e4bc6febe20e684b3aef32926ee9b9e", hashes[:2]),
        ("fr", "2026-09-21-runtime-third-slice.json"): (
            theme / filename,
            "0f0b2a14ca70c6fdc7355115e5aff7a5f31375a92f5b3d160420a456aa1c1554", hashes[2:10]),
        ("fr", "2026-09-21-runtime-fourth-slice.json"): (
            evidence / "615_requested_popup_animation_2026-10-07/original-runtime-reviews-r1"
            / "docs/i18n/reviewed/runtime/fr" / filename,
            "b754f3e4e4232bf0bed1ead8e10ccb8f8c825281fcada7bd5d24c3414dbe043a", hashes[10:]),
    }
    if (language, filename) not in specifications:
        return set()
    archive, digest, expected = specifications[(language, filename)]
    original_bytes = archive.read_bytes()
    assert hashlib.sha256(original_bytes).hexdigest() == digest
    original = json.loads(original_bytes)["records"]
    reason = (
        "2026-10-15, item 700: first-run setup no longer offers AI-provider or GitHub sign-in, "
        "so the install-and-sign-in panel that showed this caption "
        "(spacr/qt/widgets/cli_setup_panel.py) left the package."
    )
    document = json.loads((folder / filename).read_text(encoding="utf-8"))
    entries = [row for row in document.get("retired_records", []) if row.get("reason") == reason]
    assert all(set(row) == {"reason", "record"} for row in entries)
    records = [row["record"] for row in entries]
    assert len(records) == len(set(expected))
    assert {row["source_sha256"] for row in records} == set(expected)
    assert records == [row for row in original if row["source_sha256"] in expected]
    assert all(row["table"] == "ui" and row["key"] == row["source"]
               and hashlib.sha256(row["source"].encode()).hexdigest() == row["source_sha256"]
               for row in records)
    sources = {row["source"] for row in records}
    assert len(sources) == len(records)
    assert not sources & {row["source"] for row in document["records"]}
    return sources


def _automatic_report_retirements(language, filename, original):
    """Bind the replaced report captions to their exact preserved records."""
    import ast

    filenames = {
        "sv": "2026-09-21-runtime-ui-refresh.json",
        "fr": "2026-09-21-runtime-fourth-slice.json",
    }
    if filenames.get(language) != filename:
        return set()
    relative = f"docs/i18n/reviewed/runtime/{language}/{filename}"
    archive = ROOT / "docs/i18n/retired/runtime/2026-10-23-central-errors-and-ai-choice.json"
    entries = [entry for entry in json.loads(archive.read_text())["records"]
               if entry["path"] == relative]
    assert len(entries) == 1
    entry = entries[0]
    assert set(entry) == {"path", "reason", "records"}
    assert entry["reason"] == (
        "2026-10-23: retire the replaced automatic-error-reporting presentation; "
        "preserve its exact original source-bound translation.")
    expected = {
        "98eacfbd1ef751932ad138775f1813a8260e120cec8469650346bf6f850559f7",
        "425bf7825eca02541b5497dcd8c2d87ec8ad78d79a4f49f98450f6e72e1a425b",
    }
    records = entry["records"]
    assert len(records) == 2
    assert {row["source_sha256"] for row in records} == expected
    assert records == [row for row in original if row["source_sha256"] in expected]
    assert all(row["table"] == "ui" and row["key"] == row["source"]
               and hashlib.sha256(row["source"].encode()).hexdigest() == row["source_sha256"]
               for row in records)
    sources = {row["source"] for row in records}
    current = json.loads((ROOT / relative).read_text())["records"]
    assert not sources & {row["source"] for row in current}
    tree = ast.parse((ROOT / "spacr/qt/i18n_catalogs/en.py").read_text())
    ui = next(node.value for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "UI_SOURCES"
                      for target in node.targets))
    assert isinstance(ui, ast.Call) and isinstance(ui.func, ast.Name) and ui.func.id == "frozenset"
    assert not sources & set(ast.literal_eval(ui.args[0]))
    return sources


def _requested_detection_retirements(language, filename):
    """Validate the preserved October 7 records against their exact originals."""
    from build_i18n_catalogs import reviewed_runtime_translations

    evidence = ROOT / 'features/data/615_requested_popup_animation_2026-10-07'
    originals = evidence / 'original-runtime-reviews-r1'
    receipt = originals / 'receipt.json'
    assert hashlib.sha256(receipt.read_bytes()).hexdigest() == (
        '3c795473005f52251d3197dc8c45be8e41d8b787bbf60de867090cd00318cc53')
    relative = f'docs/i18n/reviewed/runtime/{language}/{filename}'
    entries = [entry for entry in json.loads(receipt.read_text(encoding="utf-8"))
               if entry['file'] == relative]
    # 2026-10-10: records whose English source changed when
    # toxoplasma_plaque_v3 became the default are moved to retired_records with
    # a dated reason; they count as retired here.
    current_doc = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    later = [row for row in current_doc.get('retired_records', [])
             if str(row.get('retired', '')).startswith('2026-10-10')]
    later_sources = {row['source'] for row in later}
    assert not later_sources & reviewed_runtime_translations(language).keys()
    if not entries:
        return later_sources
    assert len(entries) == 1
    entry = entries[0]
    archived = originals / relative
    assert hashlib.sha256(archived.read_bytes()).hexdigest() == entry['original_sha256']
    original = json.loads(archived.read_text(encoding="utf-8"))['records']
    current = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    retired = [row for row in original if row['key'] in entry['keys']]
    assert len(retired) == len(entry['keys'])
    assert all(hashlib.sha256(row['source'].encode()).hexdigest()
               == row['source_sha256'] for row in retired)
    # 2026-10-16, item 707: Make Masks layout captions retired the same way.
    # 2026-10-17, item 678: factual app-text rewrite retired the same way.
    n678 = {row['source'] for row in current.get('retired_records', [])
            if str(row.get('retired', '')).startswith(
                ('2026-10-11', '2026-10-16', '2026-10-17', TUTORIAL_RENDERER_RETIRED))}
    published = _published_panel_retirements(language, filename)
    automatic_reports = _automatic_report_retirements(language, filename, original)
    assert current['records'] == [row for row in original if row not in retired
                                  and row['source'] not in later_sources | n678 | published | automatic_reports]
    assert all(any(item.get('date') == '2026-10-07' and item.get('record') == row
                   for item in current.get('retired_records', [])) for row in retired)
    values = {row['source'] for row in retired}
    assert not values & reviewed_runtime_translations(language).keys()
    return values | later_sources | automatic_reports


def _figure_export_retirements(language, filename):
    """Verify the two published factual-prose retirements against frozen bytes."""
    if filename != "2026-09-15-integration-review.json":
        return set()
    from build_i18n_catalogs import reviewed_runtime_translations

    folder = ROOT / "docs/i18n/reviewed/runtime" / language
    archive = folder / "archive/2026-10-15-figure-export-prose" / filename
    archive_hashes = {'fr': '30e2bd2724ef34c3cb52e533fe6f2b280f926b83599b99e91b95f07d41fbdd08', 'sv': '297c6f8f924c71ec2bf1a2e84b71c3a819bdf61318c18779a509adb15d456526'}
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == archive_hashes[language]
    original = json.loads(archive.read_text(encoding="utf-8"))["records"]
    reason = (
        "2026-10-15: exact old source is no longer canonical after N678 "
        "figure-export factual corrections c1be6383a7c; superseded by "
        "2026-10-15-figure-export-factual.json. Original bytes archived beside this review.")
    retired = [row for row in json.loads((folder / filename).read_text(encoding="utf-8"))
               .get("retired_records", []) if row.get("retired") == reason]
    expected = {
        "8d0d775ecb20bc5c338a5999a5ab009eebbc708d27dd60f6732895ac953a638b",
        "8f9108d0acb7994423e04af9b8058bfa92eb76b6d570e3b2e842df3b9160db3d",
    }
    assert len(retired) == len(expected) == 2
    assert {row["source_sha256"] for row in retired} == expected
    for row in retired:
        assert hashlib.sha256(row["source"].encode()).hexdigest() == row["source_sha256"]
        assert {key: value for key, value in row.items() if key != "retired"} in original
    sources = {row["source"] for row in retired}
    assert not sources & reviewed_runtime_translations(language).keys()
    return sources


def _theme_cohort_retirements(language, filename):
    """Prove the exact source retirements without changing historical pins."""
    hashes = {
        "2026-09-15-integration-review.json":
            "ecf8d5190cb2844a1a90174e483138118389e7e9f0d67c99282adfb7f09203e1",
        "2026-09-21-runtime-ui-refresh.json":
            "87207bbab69e5a41953bdd89ebdac39d93f4c990e26a5b7ca8a0a715c58bbcf1",
        "2026-09-21-runtime-third-slice.json":
            "87207bbab69e5a41953bdd89ebdac39d93f4c990e26a5b7ca8a0a715c58bbcf1",
    }
    requested = _requested_detection_retirements(language, filename)
    if filename not in hashes:
        return requested
    folder = ROOT / "docs/i18n/reviewed/runtime" / language
    original = json.loads((folder / "archive/2026-10-06-home-cell-dino-save-themes"
                           / filename).read_text(encoding="utf-8"))["records"]
    current = json.loads((folder / filename).read_text(encoding="utf-8"))["records"]
    retired_hashes = {hashes[filename]}
    if filename == "2026-09-15-integration-review.json":
        retired_hashes.add(
            "ab388434f351b32a1a3bd22c13c65396a1c054d164983bd5fc2683dc718f85a4")
    retired = [row for row in original if row["source_sha256"] in retired_hashes]
    assert len(retired) == len(retired_hashes)
    assert {row["source_sha256"] for row in retired} == retired_hashes
    assert all(hashlib.sha256(row["source"].encode()).hexdigest()
               == row["source_sha256"] for row in retired)
    # 2026-10-11: records whose English the N678 factual review rewrote are
    # moved to retired_records; they leave the live list but are not part of
    # this cohort's count.
    n678 = {row['source'] for row in json.loads((folder / filename).read_text(encoding="utf-8"))
            .get('retired_records', [])
            if str(row.get('retired', '')).startswith(
                ('2026-10-11', '2026-10-16', '2026-10-17', TUTORIAL_RENDERER_RETIRED))}
    figure_retired = _figure_export_retirements(language, filename)
    published = _published_panel_retirements(language, filename)
    assert current == [row for row in original if row not in retired
                       and row['source'] not in requested | n678 | figure_retired | published]
    return {row["source"] for row in retired} | requested | figure_retired


def _with_training_sample_replacements(document, language, filename):
    archive = ROOT / "features/data/450_451_retired_runtime_review_2026-09-27" / language / filename
    if not archive.exists():
        return document
    original = json.loads(archive.read_text(encoding="utf-8"))["records"]
    retained = document["records"]
    # 600b, 2026-09-29: Make Masks' Features button was removed, so its
    # tooltip's reviewed record was deleted (key gone); it is not one of the
    # training-sample replacements below.
    features = {row for row in (json.dumps(r, sort_keys=True) for r in original)
                if json.loads(row)["source"] in _RETIRED_BY_600B}
    features = [json.loads(row) for row in features]
    original = [row for row in original if row not in features]
    theme_retired = _theme_cohort_retirements(language, filename)
    # 2026-10-11: N678 factual-review retirements are not training-sample
    # replacements; they sit under retired_records with their reason.
    n678 = {row["source"] for row in document.get("retired_records", [])
            if str(row.get("retired", "")).startswith(("2026-10-11",
                                                       "2026-10-16", "2026-10-17"))}
    published = _published_panel_retirements(language, filename)
    original = [row for row in original
                if row["source"] not in theme_retired | n678 | published]
    retired = [row for row in original if row not in retained]
    assert len(retired) == (2 if filename == "2026-09-21-runtime-ui-refresh.json" else 1)
    replacements = {
        "Ten fields of the dataset a published model was trained on, with the masks it was taught. They open for editing, so what you see is what the model saw.":
            "A sample of a published model's training dataset. Sample sizes vary by dataset. Masks are included where available.",
        "{name}: {count} fields and the masks the model was trained on":
            "{name}: {count} example images ready",
        "Download ten example fields for Plaque Analysis and point src at them. Two sets to choose from: segmented plaque fields, which is what the plaque model was trained on, or whole plate figures, which is what the pipeline takes. Cached after the first download.":
            "Download example data for Plaque Analysis and point src at it. Choose segmented plaque fields or whole plate figures. Sample sizes vary by dataset. Cached after the first download.",
    }
    assert {row["source"] for row in retired} <= replacements.keys()
    assert all(row in document["retired_records"] for row in retired)
    assert retained == [row for row in original if row not in retired]
    replacement = json.loads((ROOT / "docs/i18n/reviewed/runtime" / language /
                              "2026-09-27-training-samples.json").read_text(encoding="utf-8"))["records"]
    assert len(replacement) == 3
    assert {row["source"] for row in replacement} == set(replacements.values())
    from build_i18n_catalogs import reviewed_runtime_translations
    reviewed = reviewed_runtime_translations(language)
    assert not {row["source"] for row in retired} & reviewed.keys()
    assert all(reviewed[row["source"]] == row["translation"] for row in replacement)
    wanted = {replacements[row["source"]] for row in retired}
    replacement = [row for row in replacement if row["source"] in wanted]
    assert len(retained + replacement) == len(original)
    return {**document, "records": retained + replacement}


#: The three reviewed files 316 added on 2026-09-28 for inherited alpha
#: captions (2eee4bf3e, c356d4c0f): 8 QC-classifier, 11 SAM2/virtual-staining
#: and 12 distinct counterfactual/database sources, none of them shared with
#: an older file. The per-file pins below predate them, so each test takes
#: these out first and checks them here, whole.
INHERITED_2026_09_28 = ("2026-09-28-qc-classifier.json",
                        "2026-09-28-sam2-virtual-staining.json",
                        "2026-09-28-counterfactual-databases.json")


def _inherited_2026_09_28_sources(language: str, reviewed) -> set[str]:
    sources = set()
    for name in INHERITED_2026_09_28:
        document = json.loads((ROOT / "docs/i18n/reviewed/runtime" / language
                               / name).read_text(encoding="utf-8"))
        sources |= {record["source"] for record in document["records"]}
    # 31 -> 30 on 2026-09-29: the "Measurement Backend (Alpha)" caption was
    # renamed to its "α" form (591-597); the old record was deleted.
    # 30 -> 29 on 2026-10-03 (N615): the virtual-staining channel prompt
    # gained its pix2pix hint; the old record moved to retired_records.
    assert len(sources) == 29
    assert sources <= reviewed.keys()
    return sources


def _runtime_debt_sources(language: str, reviewed: dict[str, str], expected: int) -> set[str]:
    """The 2026-09-25 runtime translation debt (instruction 316), one cohort.

    Written and technically reviewed by AI against the English source, with no
    native-speaker signoff, in 2026-09-25-runtime-debt-*.json. Every source
    was missing from the catalog before, so the cohort shares no source with
    any earlier record and the older counts below hold once it is subtracted.
    """
    folder = ROOT / "docs/i18n/reviewed/runtime" / language
    paths = sorted(folder.glob("2026-09-25-runtime-debt-*.json"))
    first = [record for path in paths if "-second-pass-" not in path.name
             for record in json.loads(path.read_text(encoding="utf-8"))["records"]]
    # The second pass covers sources added on nightly after the first pass
    # was prepared (items 509-530); it is optional here so a locale can land
    # its first pass alone, and it must not overlap the first.
    second = [record for path in paths if "-second-pass-" in path.name
              for record in json.loads(path.read_text(encoding="utf-8"))["records"]]
    sources = {record["source"] for record in first}
    retired = set().union(*(_requested_detection_retirements(language, path.name)
                            for path in paths if '-second-pass-' not in path.name))
    assert len(first) == len(sources)
    assert len(sources) + len(retired) == expected
    later = {record["source"] for record in second}
    assert len(second) == len(later) and not later & sources
    sources |= later
    # The third pass (2026-09-26) covers what nightly added or reworded after
    # the second: items 426, 474, 511, 523, 528, 529, 531 and 533, and the
    # shortened ops_spot_detector and cam_type tooltips (the stale
    # ops_spot_detector second-pass record was deleted, as the loader
    # directs). It is subtracted as one more slice of the same cohort, so
    # the older counts are unchanged; every source was pending before, so
    # it overlaps neither earlier pass.
    third = [record for path in sorted(folder.glob("2026-09-26-runtime-debt-third-pass-*.json"))
             for record in json.loads(path.read_text(encoding="utf-8"))["records"]]
    latest = {record["source"] for record in third}
    assert len(third) == len(latest) and not latest & sources
    sources |= latest
    # The fourth pass (2026-09-26, CI run 36276973443) is the same kind of
    # slice: the captions 316 owed in _AWAITING_CATALOG_REBUILD (alpha items
    # 541, 544/573, 545, 548, 551-553, 555, 570 and 493's GPU controls) and
    # the tooltips whose English moved under them (percentiles, fill_in,
    # enhance_clahe, timelapse_mode, Make Masks' CLAHE caption). The stale
    # records those edits left were deleted, as the loader directs.
    fourth = [record for path in sorted(folder.glob("2026-09-26-runtime-debt-fourth-pass-*.json"))
              for record in json.loads(path.read_text(encoding="utf-8"))["records"]]
    newest = {record["source"] for record in fourth}
    assert len(fourth) == len(newest) and not newest & sources
    sources |= newest
    # The fifth pass (2026-09-26) is one more slice: items 536, 550, 577 and
    # the profiling and cell-cycle settings that reached nightly after the
    # fourth. Every source was pending, so it overlaps no earlier pass.
    fifth = [record for path in sorted(folder.glob("2026-09-26-runtime-debt-fifth-pass-*.json"))
             for record in json.loads(path.read_text(encoding="utf-8"))["records"]]
    latest5 = {record["source"] for record in fifth}
    assert len(fifth) == len(latest5) and not latest5 & sources
    sources |= latest5
    # The sixth pass (2026-09-27) is one more slice: items 539, 571, 579,
    # 582 and the other captions nightly added after the fifth.
    sixth = [record for path in sorted(folder.glob("2026-09-27-runtime-debt-sixth-pass-*.json"))
             for record in json.loads(path.read_text(encoding="utf-8"))["records"]]
    latest6 = {record["source"] for record in sixth}
    assert len(sixth) == len(latest6) and not latest6 & sources
    sources |= latest6
    # Direct Codex-reviewed delta: 129 new sources plus11 extraction repairs,
    # minus the Spotiflow identity; Hindi also resolves35 historical fallbacks.
    seventh = json.loads((folder / "2026-09-27-runtime-codex-delta.json").read_text(encoding="utf-8"))["records"]
    latest7 = {record["source"] for record in seventh}
    # 139 -> 136 (hi 174 -> 171) on 2026-09-29: items 591-597 renamed the
    # Event Detection, GPU Measurement and Segmentation Robustness categories
    # to their "α" captions, so those three records were deleted (key gone).
    # 136 -> 133 (hi 171 -> 168) on 2026-10-02 (item 43): 603423d0a moved
    # three records whose source text changed to the file's retired_records.
    # 133 -> 132 (hi 168 -> 167) on 2026-10-03 (N615): the measure_gpu
    # tooltip gained its cuCIM sentence, so its record moved to
    # retired_records; the current wording is in the alpha-batch file.
    # 2026-10-05: the watch_folder microscope-feedback tooltip was rewritten.
    original = json.loads((folder / "archive/2026-10-06-subcell-rybg/"
                           "2026-09-27-runtime-codex-delta.json").read_text(encoding="utf-8"))["records"]
    retired_hashes = {
        "d25bb244df4fcf08909bd1d860d55646aa14346420a9fc0dcace7c696cbce162",
        "f7770d466133e7f859c6a59b6865c566414a2190f4e70670291a2b3a0f194478",
        "a2aa5f4df3c6ae4f4615b6e4c4fcd3fb3c24f7e7207120071239dc2e5f0d2527",
    }
    retired = [record for record in original
               if record["source_sha256"] in retired_hashes]
    assert len(retired) == 3
    assert {record["source_sha256"] for record in retired} == retired_hashes
    assert all(hashlib.sha256(record["source"].encode()).hexdigest()
               == record["source_sha256"] for record in retired)
    assert seventh == [record for record in original if record not in retired]
    assert len(original) == len({record["source"] for record in original}) == (
        166 if language == "hi" else 131)
    assert len(seventh) == len(latest7) == len(original) - len(retired)
    assert not {record["source"] for record in retired} & reviewed.keys()
    replacements = json.loads((folder / "2026-10-06-home-cell-dino-save-themes.json"
                               ).read_text(encoding="utf-8"))["records"]
    replacement = [record for record in replacements if record["source"].startswith(
        "A model trained on microscopy rather than photographs.")]
    assert len(replacement) == 1
    assert "local official checkpoint, declared SHA-256" in replacement[0]["source"]
    assert reviewed[replacement[0]["source"]] == replacement[0]["translation"]
    from tools.build_i18n_catalogs import canonical_sources

    diameter_reviews = json.loads((folder / "2026-10-14-magnification-default-and-auto-diameter.json").read_text(encoding="utf-8"))["records"]
    diameter = [record for record in diameter_reviews
                if record["table"] == "setting_tooltips"
                and record["key"] == "robustness_diameter_factors"]
    assert len(diameter) == 1
    diameter = diameter[0]
    assert diameter["source_sha256"] == "872bf102684298902ddc7125086b211aa1147004ac597dddf106fdf2c5f73dc1"
    assert hashlib.sha256(diameter["source"].encode()).hexdigest() == diameter["source_sha256"]
    assert reviewed[diameter["source"]] == diameter["translation"]
    current_tooltips = canonical_sources()["setting_tooltips"]
    assert current_tooltips["robustness_diameter_factors"] == diameter["source"]
    current_timelapse = current_tooltips["timelapse_events"]
    assert current_timelapse.startswith("After the run, detect events on every tracked object")
    assert "optional videomae encoder" in current_timelapse
    assert current_timelapse in reviewed
    assert not latest7 & sources
    sources |= latest7
    discovery = json.loads((folder / "2026-09-27-gpu-discovery.json").read_text(encoding="utf-8"))["records"]
    discovery_sources = {record["source"] for record in discovery}
    assert len(discovery) == 1
    assert discovery_sources == {"Checking compatible GPUs…"}
    assert not discovery_sources & sources
    sources |= discovery_sources
    # The 2026-09-29 seventh runtime pass: the renamed "α" categories, the
    # channel-sort and consolidate dialogs, Noise2Void and the new category
    # help (items 591-597), one more slice that overlaps nothing earlier.
    pass7 = [record for path in sorted(folder.glob("2026-09-29-runtime-debt-seventh-pass-*.json"))
             for record in json.loads(path.read_text(encoding="utf-8"))["records"]]
    latest_p7 = {record["source"] for record in pass7}
    assert len(pass7) == len(latest_p7) and not latest_p7 & sources
    sources |= latest_p7
    # 2026-09-29 (474): the eight organism tile usage notes ("Opens ..."),
    # which entered the runtime inventory only now; another disjoint slice.
    notes = json.loads((folder / "2026-09-29-organism-workflow-notes.json")
                       .read_text(encoding="utf-8"))["records"]
    note_sources = {record["source"] for record in notes}
    assert len(notes) == len(note_sources) == 8 and not note_sources & sources
    sources |= note_sources
    # The 2026-09-30 eighth runtime pass: the Organize for Measure popup,
    # its regex teaching and drop-classification dialogs and the popup
    # table's view options (items 600 and 592); another disjoint slice.
    pass8 = [record for path in sorted(folder.glob("2026-09-30-runtime-debt-eighth-pass-*.json"))
             for record in json.loads(path.read_text(encoding="utf-8"))["records"]]
    latest_p8 = {record["source"] for record in pass8}
    assert len(pass8) == len(latest_p8) and not latest_p8 & sources
    sources |= latest_p8
    # The 2026-09-30 ninth runtime pass: the colony_detector setting label
    # and tooltip (item 542); another disjoint slice.
    pass9 = [record for path in sorted(folder.glob("2026-09-30-runtime-debt-ninth-pass-*.json"))
             for record in json.loads(path.read_text(encoding="utf-8"))["records"]]
    latest_p9 = {record["source"] for record in pass9}
    assert len(pass9) == len(latest_p9) and not latest_p9 & sources
    sources |= latest_p9
    # The 2026-09-30 tenth runtime pass: item 598's contribute-dialog
    # captions (the third-pass record of the old linked thank-you was retired
    # in place) and the Make Masks button re-layout (Upload data, the Load
    # test data menu, the Uncertainty setting).
    pass10 = [record for path in sorted(folder.glob("2026-09-30-runtime-debt-tenth-pass-*.json"))
              for record in json.loads(path.read_text(encoding="utf-8"))["records"]]
    latest_p10 = {record["source"] for record in pass10}
    assert len(pass10) == len(latest_p10) and not latest_p10 & sources
    sources |= latest_p10
    # The 2026-09-30 eleventh runtime pass, first slice: the confluency
    # tooltips item 536 rewrote (their fourth-pass records retired in place)
    # and item 603's tooltip-delay row; disjoint from every earlier slice.
    pass11a = [record for path in sorted(folder.glob("2026-09-30-runtime-debt-eleventh-pass-*.json"))
               for record in json.loads(path.read_text(encoding="utf-8"))["records"]]
    latest_p11a = {record["source"] for record in pass11a}
    assert len(pass11a) == len(latest_p11a) and not latest_p11a & sources
    sources |= latest_p11a
    # The 2026-09-30 eleventh runtime pass: meaning fixes for machine-
    # translated names (train, model zoo, batch, run, seed, hit, well, ...)
    # that had no reviewed record before; another disjoint slice.
    pass11 = [record for path in sorted(folder.glob("2026-09-30-runtime-meaning-fixes-eleventh-pass*.json"))
              for record in json.loads(path.read_text(encoding="utf-8"))["records"]]
    latest_p11 = {record["source"] for record in pass11}
    assert len(pass11) == len(latest_p11) and not latest_p11 & sources
    sources |= latest_p11
    # The 2026-10-01 twelfth runtime pass: the AI meaning review of setting
    # labels and tooltips that had only a machine translation; disjoint.
    pass12 = [record for path in sorted(folder.glob("2026-10-01-runtime-review-twelfth-pass-*.json"))
              for record in json.loads(path.read_text(encoding="utf-8"))["records"]]
    latest_p12 = {record["source"] for record in pass12}
    assert len(pass12) == len(latest_p12) and not latest_p12 & sources
    sources |= latest_p12
    # Feature batches reviewed from 2026-10-01 on (colony preview, lineage,
    # table controls, calibration gain, figure integrity, ...): every other
    # dated file in the folder, each one more disjoint slice of new sources.
    batches = [record for path in sorted(folder.glob("2026-1[0-2]-*.json"))
               if "-runtime-review-twelfth-pass-" not in path.name
               for record in json.loads(path.read_text(encoding="utf-8"))["records"]]
    latest_batches = {record["source"] for record in batches}
    repeated_sources = {
        "Batch size": "8cfe32a04155c2ea30704f9c105e6ff553fdb51ad8e1b153ab4296b554c97f7c",
        "Learning rate": "f033c501afe798cdcd46e31cc25ca5b73ee2ad88a065eed69ae056a776f7175b",
        "Train": "e99c90db87c0e400832edcedcc1b14e33cff7559988f4d7032337e17401b2945",
        # 2026-10-16 (item 705): the Measure 3D images dialog binds two
        # setting labels as plain ui captions, with the same reviewed wording.
        "Object type": "46e7c345a4106f8dba31dee33a1559a12f24d16a1678241d55e84ae55c107d81",
        "Save measurements": "e8ac777599eba6295d02d6d7b4ba13d74eecc2b61383369c8bf9c99134423063",
        # 2026-10-25 (item 725): the Overview tile-size slider binds the reviewed
        # image_size setting label as a plain ui caption with the same wording.
        "Image size": "80bb38dca36b0439367670908cfe04711b4e9d5dfabc789bf4dd8f21935cd4f9",
    }
    assert latest_batches & sources == repeated_sources.keys()
    current_files = {
        "Object type": "2026-10-16-705-ui-arrival-bindings.json",
        "Save measurements": "2026-10-16-705-ui-arrival-bindings.json",
        "Image size": "2026-10-25-overview-round-two-cq1-psf-and-measurements.json",
    }
    previous_keys = {"Batch size": "batch_size", "Learning rate": "learning_rate", "Train": "train",
                     "Object type": "object_type", "Save measurements": "save_measurements",
                     "Image size": "image_size"}
    for source, source_hash in repeated_sources.items():
        current_file = current_files.get(source, "2026-10-15-f561-classifier-ui-runtime-home-b.json")
        current = [row for row in json.loads((folder / current_file).read_text(encoding="utf-8"))["records"]
                   if row["source"] == source]
        assert len(current) == 1
        current = current[0]
        assert current["table"] == "ui" and current["key"] == source
        assert current["source_sha256"] == source_hash
        assert hashlib.sha256(source.encode()).hexdigest() == source_hash
        previous_file = (
            "2026-09-30-runtime-meaning-fixes-eleventh-pass.json"
            if source == "Train" or (language == "sv" and source == "Batch size")
            else "2026-10-01-runtime-review-twelfth-pass-labels.json"
        )
        previous = [row for row in json.loads((folder / previous_file).read_text(encoding="utf-8"))["records"]
                    if row["source"] == source]
        assert len(previous) == 1
        previous = previous[0]
        assert previous["table"] == "setting_labels"
        assert previous["key"] == previous_keys[source]
        assert previous["source_sha256"] == source_hash
        assert previous["translation"] == current["translation"] == reviewed[source]
        assert [row for row in batches if row["source"] == source] == [current]
    sources |= latest_batches
    assert sources <= reviewed.keys()
    return sources


def test_swedish_example_abbreviation_is_not_a_dotted_identifier() -> None:
    from build_i18n_catalogs import _syntax_preserved

    source = "Choose a column, e.g. 'plateID', from measurements.db."
    translated = "Välj en kolumn, t.ex. 'plateID', från measurements.db."

    assert _syntax_preserved(source, translated)
    assert not _syntax_preserved(
        source,
        translated.replace("measurements.db", "measurements.database"),
    )


def test_swedish_reviewed_runtime_text_is_source_bound_and_gate_clean() -> None:
    from build_i18n_catalogs import (
        _looks_translatable,
        _translation_rejection_reasons,
        canonical_sources,
        reviewed_runtime_translations,
    )

    reviewed = reviewed_runtime_translations("sv")
    published_retired = _published_panel_retirements("sv", "2026-09-21-runtime-ui-refresh.json")
    assert len(published_retired) == 15
    assert not published_retired & reviewed.keys()
    base_retired = _theme_cohort_retirements("sv", "2026-09-15-integration-review.json")
    refresh_retired = _theme_cohort_retirements("sv", "2026-09-21-runtime-ui-refresh.json")
    cohort_retired = base_retired | refresh_retired
    assert len(base_retired) == 4 and len(refresh_retired) == 4
    assert not cohort_retired & reviewed.keys()
    # +269 distinct UI/category sources, with three OPS descriptions shared
    # between both tables (272 records). The detection-panel consolidation
    # retired five source captions, preserving their records in the archive.
    ui_refresh = json.loads((ROOT / "docs/i18n/reviewed/runtime/sv/"
                              "2026-09-21-runtime-ui-refresh.json").read_text(encoding="utf-8"))
    ui_refresh = _with_training_sample_replacements(
        ui_refresh, "sv", "2026-09-21-runtime-ui-refresh.json")
    ui_sources = {record["source"] for record in ui_refresh["records"]}
    # Item 511 retired four Make Masks filter captions (the fixed bounds'
    # button, ledger, card and placeholder help): 263 -> 259, 260 -> 256.
    # Instruction 316 then retired the 17 setup and sign-in captions that
    # moved to _ROWS (kept under retired_records): 259 -> 242, 256 -> 239.
    # 600b, 2026-09-29: the Features button's tooltip was retired (Make
    # Masks' button removed): 242 -> 241, 239 -> 238.
    # 2026-10-11: 241 -> 240, one record retired by the N678 factual review.
    # 2026-10-16, item 707: the Make Masks Settings button's tooltip was
    # retired (the button is gone): 240 -> 239, 237 -> 236.
    # 2026-10-17, item 678: eight records whose English the Home app-text
    # pass rewrote moved to retired_records: 239 -> 231, 236 -> 228.
    assert len(ui_refresh["records"]) + len(refresh_retired) + len(published_retired) == 231  # Three old threshold/histogram reviews archived.
    assert len(ui_sources) + len(refresh_retired) + len(published_retired) == 228  # 238 -> 237 on 2026-10-11 (N678); 236 on 2026-10-16 (707); 228 on 2026-10-17 (678 app text)
    assert ui_sources <= reviewed.keys()
    all_reviewed = reviewed
    examples = json.loads((ROOT / "docs/i18n/reviewed/runtime/sv/"
                            "2026-09-21-assay-and-plate-examples.json").read_text(encoding="utf-8"))
    example_sources = {record["source"] for record in examples["records"]}
    assert len(example_sources) == 7
    assert example_sources <= all_reviewed.keys()
    assert not example_sources & ui_sources
    previews = json.loads((ROOT / "docs/i18n/reviewed/runtime/sv/"
                            "2026-09-21-preview-refresh.json").read_text(encoding="utf-8"))
    preview_sources = {record["source"] for record in previews["records"]}
    assert len(preview_sources) == 5
    assert preview_sources <= all_reviewed.keys()
    assert not preview_sources & example_sources
    normalized = json.loads((ROOT / "docs/i18n/reviewed/runtime/sv/"
                              "2026-09-21-normalized-detection.json").read_text(encoding="utf-8"))
    normalized_sources = {record["source"] for record in normalized["records"]}
    assert len(normalized_sources) == 5
    assert normalized_sources <= all_reviewed.keys()
    assert not normalized_sources & (example_sources | preview_sources)
    download_sources = _new_download_sources("sv", all_reviewed)
    subsequent_sources = _subsequent_review_sources("sv", all_reviewed)
    # +720/-0 on 2026-09-25: the runtime translation debt cohort (316).
    # 720 -> 712 on 2026-09-25 (474): eight organism-page paragraphs were
    # rewritten when twenty proposals became live or Coming soon tiles.
    # 712 -> 711 on 2026-09-26 (316 fourth pass): Make Masks' CLAHE caption
    # now says "saturating" where it said "blowing out", so its first-pass
    # record left 2026-09-25-runtime-debt-messages.json; the new wording is
    # a fourth-pass record.
    # 711 -> 710 on 2026-09-28: 595 removed the "Measurement Features"
    # caption, so its record left 2026-09-25-runtime-debt-captions.json.
    debt_sources = _runtime_debt_sources("sv", all_reviewed, 699)  # 706 -> 699 on 2026-10-12: the seven narration captions of the deleted spacr-tutorial renderer retired; 707 -> 706 on 2026-10-11: one record retired by the N678 factual review; 708 -> 707 on 2026-10-04 (656/657): the settings-template file filter gained Settings CSV, its old record retired in place; 710 -> 709 on 2026-09-29: the "Point Spread Function" category caption was renamed (591-597); 709 -> 708 on 2026-10-02 (item 43): a retired workflow phrase (603423d0a)
    assert not debt_sources & (ui_sources | example_sources | preview_sources | normalized_sources | download_sources | subsequent_sources)
    inherited_sources = _inherited_2026_09_28_sources("sv", all_reviewed)
    assert not inherited_sources & (debt_sources | ui_sources | download_sources | subsequent_sources)
    older_all_sources = all_reviewed.keys() - download_sources - subsequent_sources - debt_sources - inherited_sources
    reviewed = {source: value for source, value in all_reviewed.items()
                if source not in ui_sources | example_sources | preview_sources | normalized_sources | download_sources | subsequent_sources | debt_sources | inherited_sources}
    sources = canonical_sources()
    current_values = set(sources["setting_labels"].values())
    current_values.update(sources["setting_tooltips"].values())
    current_values.update(sources["ui"])
    # EVERY TABLE A RECORD MAY BIND TO, not three of the five. The loader
    # validates module_summaries and categories records against
    # canonical_sources() exactly as it does the others, but this set left
    # them out, which went unnoticed while no Swedish or French record used
    # them. cdbb41cbd's Dose-Response summary and runtime pass A's OPS fold
    # sentence are module_summaries records, so the set now matches the
    # loader's own tables; the assertion below is unchanged.
    current_values.update(sources["module_summaries"].values())
    current_values.update(sources["categories"])

    # THE NUMBER FOLLOWS THE EVIDENCE, not the other way round. These count
    # the records under docs/i18n/reviewed/runtime/<lang>, and they last moved
    # on 2026-09-08 for the release: Swedish 116 -> 118, the two SETTING_LABELS
    # rows the build's own QA gate reported as still exact English
    # (`arr_axes`, `cellpose_diameter`). French is unchanged at 96. The loop
    # below is the actual contract: every record must still bind to a live
    # source value and still pass the current syntax, semantic, script and
    # exact-copy gates, so a record that has drifted fails here rather than
    # being absorbed by a looser count.
    # 118 -> 134 on 2026-09-15, +16/-0: the live magnifier's captions (407),
    # written by hand as reviewed records so its first catalog build needed
    # no model. 'Cellpose' has no record: a name is kept as it is by the
    # gates, and an identity record is rejected as an exact copy.
    #
    # 134 -> 191 on 2026-09-15, +57/-0, for the 08:15 integration batch
    # (wip/integ-0815) on nightly 554dcf371:
    #    2  the `segmentation_backend` label and tooltip (404/405),
    #       2026-09-15-segmentation-backend.json
    #   55  corrections from reading every row the batch's GPU pass changed,
    #       2026-09-15-integration-review.json -- including the Cellpose 4
    #       diameter-check UI row the model left English, whose reviewed
    #       translation replaced the hand-written one first recorded in
    #       2026-09-15-integration-new.json (that file is gone)
    # PROVED BY SUBTRACTION with the loader itself: the live count minus the
    # sources in those two files is 134, they share no source with each other,
    # and neither shares one with any other file.
    #
    # 191 -> 201 on 2026-09-15, +10/-0, for item 286: the five performance
    # level tooltips and the five hardware notes, hand-written in
    # 2026-09-15-performance-levels.json. None had ever reached a translator:
    # the runtime extractor iterated the two dicts and collected their keys.
    # All ten sources are new wording, so the file shares no source with any
    # other, and the live count minus its ten is 191.
    #
    # 201 -> 219 on 2026-09-15, +21/-3, for the magnifier's border option and
    # whole-image mode (407), rebased onto nightly df1216b3f. The 21 are
    # 2026-09-15-magnifier-whole-image.json; the 3 left
    # 2026-09-15-live-magnifier.json with the card subtitle, the Size tooltip
    # and the Magnifier-button tooltip, whose English was reworded because the
    # whole-image mode made it untrue. PROVED BY SUBTRACTION with the loader:
    # 201 + 21 - 3 = 219 is the live count, the live count minus that file's
    # sources is 198 (201 - 3), and the file shares no source with any other.
    # +2 on 2026-09-15, both from 387's selectivity index on the
    # Dose-Response screen: "Host response" and its tooltip, in
    # 2026-09-15-dose-response-host-readout.json. Staged for the catalog
    # pass rather than regenerated here, as the catalog lane asked.
    # +4 more on 2026-09-15, from 387's checkerboard scoring: "Second
    # compound", its tooltip, "Bliss independence" and "Loewe
    # additivity", in 2026-09-15-dose-response-combination.json.
    # 225 -> 263 on 2026-09-15, +38/-0, for runtime pass A on nightly
    # 5a9c4563a: 11 Dose-Response terms and Cache ceiling
    # (2026-09-15-dose-response-terms.json, 2026-09-15-performance-terms.json,
    # 12), the rewritten OPS captions (2026-09-15-ops-engine-captions.json, 10),
    # "Controls" (2026-09-15-controls-experimental.json, 1) and the
    # Dose-Response grid's unrecorded captions (2026-09-15-dose-response-grid.json,
    # 15). Swedish had no record for the retired OPS wording, and the
    # segmentation_backend tooltip record was replaced in place. PROVED BY
    # SUBTRACTION with the loader: 225 + 38 = 263 is the live count, the live
    # count minus those five files' 38 sources is 225, and none of the five
    # shares a source with any other file.
    #
    # 263 -> 269 on 2026-09-15, +6/-0: Make Masks' "Load test data…" tooltip
    # and its five status strings (412), 412-make-masks-demo.json, written by
    # hand so the button's first catalog build needed no model. The live
    # count minus that file's six sources is 263, and none of the six is in
    # any other file.
    #
    # 269 -> 284 on 2026-09-15, +15/-0: the update dialog's strings and the
    # user-visible removal reasons (416),
    # 2026-09-15-update-removes-old-installs.json. Its branch never moved
    # this pin. The live count minus that file's fifteen sources is 269, and
    # none of the fifteen is in any other file.
    #
    # 284 -> 299 on 2026-09-15, +18/-3, for the combined magnifier second round
    # (417): the Otsu threshold correction, the Model zoo… button, the note
    # that the Cellpose-SAM settings drive the magnifier, the DINOCell and
    # SAMCell install lines, "{name} (not downloaded)", and five reworded
    # tooltips (Size, Mode, Sensitivity, Model, Otsu detect), in
    # 2026-09-15-magnifier-round-two-settings.json. Retired: the old Mode,
    # Sensitivity and Size wordings from the two earlier magnifier files.
    # "DINOCell" and "SAMCell" themselves are builder _IDENTITY_TEXT, not
    # records. The second branch adds six more sources for drag-to-merge (417,
    # parts 5-6) -- the "Objects added" row, its two choices and tooltip, and
    # two status lines -- 2026-09-15-magnifier-round-two-drag.json, written by
    # hand. Measured on the combined source: all eighteen sources are distinct
    # and present. Removing them leaves 281 = 284 - 3; 281 + 18 = 299.
    # 299 -> 302: three distinct Dose-Response report sources gain reviewed
    # wording (pooled EC50, selectivity index, combination-model excess).
    # None had a reviewed record before; removing these three returns 299.
    #
    # 302 -> 319 on 2026-09-20, and the number had not been checkable since
    # 2026-09-15. From that date the loader raised on the first stale record
    # it met, so every count below was pinned against a tree whose records
    # could not be loaded at all, and three commits then added records nobody
    # could count. Item 446's pass fixed the loader's input rather than the
    # loader: 144 records over nine languages pinned an English string that
    # no longer exists -- 15 sources renamed or rewritten out of spaCR by
    # 417, 419, 423 and 435 (Classical, Otsu threshold correction, the two
    # pip-install lines 423 replaced with the Model Zoo, the magnifier's Mode
    # and Model tooltips, two OPS category captions) and 5 whose English was
    # edited under an unchanged key. Retiring a record whose source is gone
    # is what 417 (f6c511cc3) and 418 (c0b2c5227) already did.
    #
    # THE SWEDISH ARITHMETIC, in raw records: 303 at the pin, +27 for 418
    # (c0b2c5227), +7 for the settings packs (a64a93e47), 364 (8619dcb9c)
    # edited without adding, = 337; this pass removes 16, leaving 321.
    # The assertion counts DISTINCT sources, and 2 of those raw records
    # repeat a source another record already carries, so 319.
    # +8/-0 on 2026-09-21: current Make Masks readouts. Subtracting
    # this file's distinct sources restores the previously verified count.
    readouts = json.loads((ROOT / "docs/i18n/reviewed/runtime/sv/"
                           "2026-09-21-make-masks-readouts.json").read_text(encoding="utf-8"))
    added_sources = {record["source"] for record in readouts["records"]}
    assert len(added_sources) == 7  # Quality already has a compact owner.
    assert added_sources <= reviewed.keys()
    background = json.loads((ROOT / "docs/i18n/reviewed/runtime/sv/"
                              "2026-09-21-organelle-background.json").read_text(encoding="utf-8"))
    background_sources = {record["source"] for record in background["records"]}
    assert len(background_sources) == 8
    assert not added_sources & background_sources
    assert background_sources <= reviewed.keys()
    samples = json.loads((ROOT / "docs/i18n/reviewed/runtime/sv/"
                           "2026-09-21-dataset-sample-counts.json").read_text(encoding="utf-8"))
    samples = _with_training_sample_replacements(
        samples, "sv", "2026-09-21-dataset-sample-counts.json")
    sample_sources = {record["source"] for record in samples["records"]}
    assert len(sample_sources) == 3
    assert sample_sources <= reviewed.keys()
    assert not sample_sources & (added_sources | background_sources)
    scientific = json.loads((ROOT / "docs/i18n/reviewed/runtime/sv/"
                              "2026-09-21-scientific-settings.json").read_text(encoding="utf-8"))
    scientific_sources = {record["source"] for record in scientific["records"]}
    assert len(scientific_sources) == 35  # 37 -> 35 on 2026-10-11: two records retired by the N678 factual review
    assert not scientific_sources & _compact_tooltip_sources("sv")
    scientific_sources |= _compact_tooltip_sources("sv")
    assert len(scientific_sources) == 37  # 39 -> 37 on 2026-10-11 (N678 retirements)
    assert scientific_sources <= reviewed.keys()
    assert not scientific_sources & (added_sources | background_sources)
    assert not sample_sources & scientific_sources
    older_sources = reviewed.keys() - scientific_sources - sample_sources
    # Four retired chrome/template sources are preserved in the September 23 archive.
    # 313 -> 306 on 2026-09-25, item 511: the maintainer retired the cell,
    # nucleus and pathogen mean-bound settings into object_filters rows, and
    # the seven sv records of their labels and tooltips were deleted from
    # 2026-09-15-mask-mean-bounds.json (the English is gone).
    # 2026-10-05: Bar alpha left the UI, so its reviewed source was retired.
    assert len(older_sources - added_sources - background_sources) + len(base_retired) == 300  # 301 -> 300 on 2026-10-17 (N703: the 2026-09-15 update prompt left the source when updates moved to one Restart prompt; archived under archive/2026-10-17-n703-retired-update-prompts).  # 316 fourth pass (2026-09-26): -1, the percentiles tooltip record left 2026-08-14-exact-final.json (its English changed).  # 304 -> 301 on 2026-10-11 (N678 retirements)
    # +8/-0: four source-bound background labels and four scientific tooltips.
    assert len(older_sources - background_sources) + len(base_retired) == 307  # 308 -> 307 on 2026-10-17 (N703: a retired per-adapter update prompt was archived).  # Item 511 retirement (2026-09-25): -7.  # 316 fourth pass (2026-09-26): -1, the percentiles tooltip record left 2026-08-14-exact-final.json (its English changed).  # 311 -> 308 on 2026-10-11 (N678 retirements)
    assert len(older_sources) + len(base_retired) == 315  # 316 -> 315 on 2026-10-17 (N703: a retired per-adapter update prompt was archived).  # Item 511 retirement (2026-09-25): -7.  # 316 fourth pass (2026-09-26): -1, the percentiles tooltip record left 2026-08-14-exact-final.json (its English changed).  # 319 -> 316 on 2026-10-11 (N678 retirements)
    assert len(reviewed.keys() - sample_sources) + len(base_retired) == 352  # 353 -> 352 on 2026-10-17 (N703: a retired per-adapter update prompt was archived).  # +39 scientific sources. Item 511 retirement (2026-09-25): -7.  # 316 fourth pass (2026-09-26): -1, the percentiles tooltip record left 2026-08-14-exact-final.json (its English changed).  # 358 -> 353 on 2026-10-11 (N678 retirements)
    assert len(reviewed) + len(base_retired) == 355  # 356 -> 355 on 2026-10-17 (N703: a retired per-adapter update prompt was archived).  # Features, Controls and Quality use compact rows. Item 511 retirement (2026-09-25): -7.  # 316 fourth pass (2026-09-26): -1, the percentiles tooltip record left 2026-08-14-exact-final.json (its English changed).  # 361 -> 356 on 2026-10-11 (N678 retirements)
    # The new panel cohort also reuses the earlier whole-field model tooltip.
    # 316 (71071b6c6) retired 17 setup and sign-in captions to _ROWS: -17 below;
    # the total also loses its sign-in-status record and a superseded psf-help record.
    assert len(older_all_sources - example_sources - preview_sources - normalized_sources) + len(cohort_retired) + len(published_retired) == 582  # 591 -> 590 on 2026-10-17 (N703: a retired per-adapter update prompt was archived).  # 707 (2026-10-16): -1, the Make Masks Settings tooltip retired.  # 600b (2026-09-29): -1, the Features tooltip retired.  # Item 511 retired four filter captions. Item 511 retirement (2026-09-25): -7.  # 316 fourth pass (2026-09-26): -1, the percentiles tooltip record left 2026-08-14-exact-final.json (its English changed).  # 598 -> 592 on 2026-10-11 (N678 retirements)  # 590 -> 582 on 2026-10-17 (item 678 Home app-text retirements)
    assert len(older_all_sources - preview_sources - normalized_sources) + len(cohort_retired) + len(published_retired) == 589  # 598 -> 597 on 2026-10-17 (N703: a retired per-adapter update prompt was archived).  # 707 (2026-10-16): -1, the Make Masks Settings tooltip retired.  # 600b (2026-09-29): -1, the Features tooltip retired.  # Item 511 retired four filter captions. Item 511 retirement (2026-09-25): -7.  # 316 fourth pass (2026-09-26): -1, the percentiles tooltip record left 2026-08-14-exact-final.json (its English changed).  # 605 -> 599 on 2026-10-11 (N678 retirements)  # 597 -> 589 on 2026-10-17 (item 678 Home app-text retirements)
    assert len(older_all_sources - normalized_sources) + len(cohort_retired) + len(published_retired) == 594  # 603 -> 602 on 2026-10-17 (N703: a retired per-adapter update prompt was archived).  # 600b (2026-09-29): -1, the Features tooltip retired.  # Item 511 retirement (2026-09-25): -7.  # 316 fourth pass (2026-09-26): -1, the percentiles tooltip record left 2026-08-14-exact-final.json (its English changed).  # 600b (2026-09-29): -1, the Features tooltip retired.  # 610 -> 604 on 2026-10-11 (N678 retirements)  # 707 (2026-10-16): 604 -> 603, Make Masks layout captions retired and added.  # 602 -> 594 on 2026-10-17 (item 678 Home app-text retirements)
    assert len(older_all_sources) + len(cohort_retired) + len(published_retired) == 599  # 608 -> 607 on 2026-10-17 (N703: a retired per-adapter update prompt was archived).  # 600b (2026-09-29): -1, the Features tooltip retired.  # Item 511 retirement (2026-09-25): -7.  # 316 fourth pass (2026-09-26): -1, the percentiles tooltip record left 2026-08-14-exact-final.json (its English changed).  # 600b (2026-09-29): -1, the Features tooltip retired.  # 615 -> 609 on 2026-10-11 (N678 retirements)  # 707 (2026-10-16): 609 -> 608, Make Masks layout captions retired and added.  # 607 -> 599 on 2026-10-17 (item 678 Home app-text retirements)
    # Item463 retired one superseded download tooltip; its full old evidence
    # and exact set difference are checked by _new_download_sources above.
    assert len(all_reviewed.keys() - subsequent_sources - debt_sources - inherited_sources) + len(cohort_retired) + len(published_retired) == 609  # 618 -> 617 on 2026-10-17 (N703: a retired per-adapter update prompt was archived).  # 600b (2026-09-29): -1, the Features tooltip retired.  # 625 -> 619 on 2026-10-11 (N678 retirements)  # 707 (2026-10-16): 619 -> 618, Make Masks layout captions retired and added.  # 617 -> 609 on 2026-10-17 (item 678 Home app-text retirements)
    assert len((all_reviewed.keys() | _superseded_enhancement_sources("sv", all_reviewed)) - debt_sources - inherited_sources) + len(cohort_retired) + len(published_retired) + len(_requested_detection_retirements('sv', '2026-09-22-detection-panel.json')) + len(_later_cohort_retirements('sv', all_reviewed)) == 1661  # 2026-10-24: -1, the old Timelapse workflow-map guidance retired when Timelapse moved under Mask's Time control.  # 1671 -> 1670 on 2026-10-17 (N703: a retired per-adapter update prompt was archived).  # 2026-10-10: -1, the plaque_model tooltip retired when toxoplasma_plaque_v3 became the default.  # 2026-10-04: -2, the withdrawn Plasmodium/Candida guide notes.  # 2026-10-02 (item 43): -2, 603423d0a retired the two later-cohort workflow phrases (778 -> 776 above).  # 2026-10-01: -2, the two FEATURES workflow-map phrases retired.  # 591-597 (2026-09-29): -1, the renamed "Cloud" category caption.  # 600b (2026-09-29): -1, the Features tooltip retired.  # 1678 -> 1672 on 2026-10-11 (N678 retirements)  # 707 (2026-10-16): 1672 -> 1671, Make Masks layout captions retired and added.  # 1670 -> 1662 on 2026-10-17 (item 678 Home app-text retirements)
    assert not published_retired & current_values
    for source, translated in all_reviewed.items():
        assert source in current_values
        assert not _translation_rejection_reasons(
            source,
            translated,
            "sv",
            force=_looks_translatable(source),
        )


def test_french_reviewed_runtime_text_is_source_bound_and_gate_clean() -> None:
    from build_i18n_catalogs import (
        _looks_translatable,
        _translation_rejection_reasons,
        canonical_sources,
        reviewed_runtime_translations,
    )

    all_reviewed = reviewed_runtime_translations("fr")
    published_slices = {name: _published_panel_retirements(
        "fr", f"2026-09-21-runtime-{name}-slice.json") for name in ("second", "third", "fourth")}
    assert {name: len(rows) for name, rows in published_slices.items()} == {
        "second": 2, "third": 8, "fourth": 5}
    published_retired = set().union(*published_slices.values())
    assert len(published_retired) == 15
    assert not published_retired & all_reviewed.keys()
    base_retired = _theme_cohort_retirements("fr", "2026-09-15-integration-review.json")
    refresh_retired = _theme_cohort_retirements("fr", "2026-09-21-runtime-third-slice.json")
    refresh_retired |= _theme_cohort_retirements("fr", "2026-09-21-runtime-fourth-slice.json")
    cohort_retired = base_retired | refresh_retired
    assert len(base_retired) == 4 and len(refresh_retired) == 4
    assert not cohort_retired & all_reviewed.keys()
    refresh = json.loads((ROOT / "docs/i18n/reviewed/runtime/fr/"
                          "2026-09-21-runtime-first-slice.json").read_text(encoding="utf-8"))
    refresh_sources = {record["source"] for record in refresh["records"]}
    # Item 511 retired one filter caption from each of the four slices, and
    # instruction 316 retired the setup and sign-in captions that moved to
    # _ROWS (2, 7, 4 and 4 per slice, kept under retired_records).
    # 2026-10-11: -3, records whose English the N678 factual review rewrote
    # (moved to retired_records with the reason).
    # 2026-10-17: -5, the item 678 Home app-text pass retired five more.
    assert len(refresh["records"]) == len(refresh_sources) == 65
    assert not refresh_sources & _compact_tooltip_sources("fr")
    refresh_sources |= _compact_tooltip_sources("fr")
    assert len(refresh_sources) == 67  # 72 -> 67 on 2026-10-17 (item 678 Home app-text retirements)
    assert refresh_sources <= all_reviewed.keys()
    second = json.loads((ROOT / "docs/i18n/reviewed/runtime/fr/"
                         "2026-09-21-runtime-second-slice.json").read_text(encoding="utf-8"))
    second = _with_training_sample_replacements(
        second, "fr", "2026-09-21-runtime-second-slice.json")
    second_sources = {record["source"] for record in second["records"]}
    # 64 -> 63 on 2026-09-29: 600b retired the Features button's tooltip.
    assert len(second["records"]) + len(published_slices["second"]) == len(second_sources) + len(published_slices["second"]) == 63
    assert second_sources <= all_reviewed.keys()
    assert not refresh_sources & second_sources
    refresh_sources |= second_sources
    # Two third-slice and three fourth-slice captions left with the old panel.
    # 2026-10-16, item 707: the third slice's "Prev image" record was
    # retired with the button's rename to Previous: 68 -> 67.
    # 2026-10-17, item 678: three fourth-slice records retired: 71 -> 68.
    for filename, expected in (("third", 67), ("fourth", 68)):
        document = json.loads((ROOT / "docs/i18n/reviewed/runtime/fr/"
                               f"2026-09-21-runtime-{filename}-slice.json").read_text(encoding="utf-8"))
        document = _with_training_sample_replacements(
            document, "fr", f"2026-09-21-runtime-{filename}-slice.json")
        added = {record["source"] for record in document["records"]}
        retired_slice = _theme_cohort_retirements(
            "fr", f"2026-09-21-runtime-{filename}-slice.json")
        assert len(document["records"]) + len(retired_slice) + len(published_slices[filename]) == len(added) + len(retired_slice) + len(published_slices[filename]) == expected
        assert added <= all_reviewed.keys()
        assert not added & refresh_sources
        refresh_sources |= added
    # 2026-10-11: -3, the N678 factual-review retirements (see the first slice).
    assert len(refresh_sources) + len(refresh_retired) + len(published_retired) == 265  # 707 (2026-10-16): -1, Prev image retired.  # 600b (2026-09-29): -1, the Features tooltip retired.  # Three superseded threshold/histogram sources.  # 273 -> 265 on 2026-10-17 (item 678 Home app-text retirements)
    actions = json.loads((ROOT / "docs/i18n/reviewed/runtime/fr/"
                          "2026-09-21-action-labels.json").read_text(encoding="utf-8"))
    action_sources = {record["source"] for record in actions["records"]}
    assert len(actions["records"]) == len(action_sources) == 5
    assert action_sources <= all_reviewed.keys()
    assert not action_sources & refresh_sources
    refresh_sources |= action_sources
    examples = json.loads((ROOT / "docs/i18n/reviewed/runtime/fr/"
                            "2026-09-21-assay-and-plate-examples.json").read_text(encoding="utf-8"))
    example_sources = {record["source"] for record in examples["records"]}
    assert len(example_sources) == 7
    assert example_sources <= all_reviewed.keys()
    previews = json.loads((ROOT / "docs/i18n/reviewed/runtime/fr/"
                            "2026-09-21-preview-refresh.json").read_text(encoding="utf-8"))
    preview_sources = {record["source"] for record in previews["records"]}
    assert len(preview_sources) == 5
    assert preview_sources <= all_reviewed.keys()
    assert not preview_sources & example_sources
    normalized = json.loads((ROOT / "docs/i18n/reviewed/runtime/fr/"
                              "2026-09-21-normalized-detection.json").read_text(encoding="utf-8"))
    normalized_sources = {record["source"] for record in normalized["records"]}
    assert len(normalized_sources) == 5
    assert normalized_sources <= all_reviewed.keys()
    assert not normalized_sources & (example_sources | preview_sources)
    download_sources = _new_download_sources("fr", all_reviewed)
    assert not refresh_sources & (download_sources | example_sources |
                                  preview_sources | normalized_sources)
    subsequent_sources = _subsequent_review_sources("fr", all_reviewed)
    # +720/-0 on 2026-09-25: the runtime translation debt cohort (316).
    # 720 -> 712 on 2026-09-25 (474): eight organism-page paragraphs were
    # rewritten when twenty proposals became live or Coming soon tiles.
    # 712 -> 711 on 2026-09-26 (316 fourth pass): Make Masks' CLAHE caption
    # now says "saturating" where it said "blowing out", so its first-pass
    # record left 2026-09-25-runtime-debt-messages.json; the new wording is
    # a fourth-pass record.
    # 711 -> 710 on 2026-09-28: 595 removed the "Measurement Features"
    # caption, so its record left 2026-09-25-runtime-debt-captions.json.
    debt_sources = _runtime_debt_sources("fr", all_reviewed, 699)  # 706 -> 699 on 2026-10-12: the seven narration captions of the deleted spacr-tutorial renderer retired; 707 -> 706 on 2026-10-11: one record retired by the N678 factual review; 708 -> 707 on 2026-10-04 (656/657): the settings-template file filter gained Settings CSV, its old record retired in place; 710 -> 709 on 2026-09-29: the "Point Spread Function" category caption was renamed (591-597); 709 -> 708 on 2026-10-02 (item 43): a retired workflow phrase (603423d0a)
    assert not debt_sources & (example_sources | preview_sources | normalized_sources | download_sources | refresh_sources | subsequent_sources)
    inherited_sources = _inherited_2026_09_28_sources("fr", all_reviewed)
    assert not inherited_sources & (debt_sources | refresh_sources | download_sources | subsequent_sources)
    older_all_sources = all_reviewed.keys() - download_sources - refresh_sources - subsequent_sources - debt_sources - inherited_sources
    reviewed = {source: value for source, value in all_reviewed.items()
                if source not in example_sources | preview_sources | normalized_sources | download_sources | refresh_sources | subsequent_sources | debt_sources | inherited_sources}
    sources = canonical_sources()
    current_values = set(sources["setting_labels"].values())
    current_values.update(sources["setting_tooltips"].values())
    current_values.update(sources["ui"])
    # EVERY TABLE A RECORD MAY BIND TO, not three of the five. The loader
    # validates module_summaries and categories records against
    # canonical_sources() exactly as it does the others, but this set left
    # them out, which went unnoticed while no Swedish or French record used
    # them. cdbb41cbd's Dose-Response summary and runtime pass A's OPS fold
    # sentence are module_summaries records, so the set now matches the
    # loader's own tables; the assertion below is unchanged.
    current_values.update(sources["module_summaries"].values())
    current_values.update(sources["categories"])

    # THE NUMBER FOLLOWS THE EVIDENCE, not the other way round. These count
    # the records under docs/i18n/reviewed/runtime/<lang>, and they last moved
    # on 2026-09-08 for the release: French 96 -> 97, the single row
    # (`meta_regex`) the build's own QA gate still reported as exact English.
    #
    # ITS FIRST DRAFT WAS REJECTED, and for the right reason: "si bien que"
    # put the ADVERB sense of "well" into a caption whose subject is the plate
    # well, which is precisely what `scientific-well-as-adverb` exists to
    # catch. Rephrased to "de sorte que".
    #
    # 97 -> 99 later the same day, +5/-3. The five are the cellprob-threshold
    # tooltips the work session's ranker found reading "gouttes" -- droplets
    # -- where the English says the threshold DROPS faint organelles. They
    # replace three records rather than adding five, because all four
    # `organelle*_cellprob_threshold` tooltips share a byte-identical English
    # source across the slots and the store keys by SOURCE: four slot-numbered
    # translations of one string is a contradiction, and it is rejected as
    # one. They now share a single target.
    #
    # The loop below is the actual contract: every record must still bind to a
    # live source value and still pass the current syntax, semantic, script
    # and exact-copy gates, so a record that has drifted fails here rather
    # than being absorbed by a looser count.
    # 99 -> 98 on 2026-09-11, +1/-2, and the number follows the evidence as
    # this note says it must.
    #
    #   GONE, both of them tooltips for settings 364 deleted: the legacy
    #   Image-UMAP crop selector ("the current workflow always joins
    #   measurement tables...") and one of the legacy compatibility fields
    #   ("current plotting paths do not read it"). A reviewed record for a
    #   source that is no longer in the catalog is not evidence about
    #   anything, and it left with the setting.
    #
    #   ADDED: 'press Escape to close'. The rebuilt catalogs had replaced a
    #   good row with "Appuyez sur Échapper pour fermer" -- the key name
    #   translated as the verb -- in eight of the nine languages, and
    #   nothing claimed the row, so the rebuild was free to. It is claimed
    #   now.
    #
    # 98 -> 99 on 2026-09-15, +1. ADDED: the `resume` setting label,
    # 'Reprendre'. Five locales rendered "Resume" as the CV noun (resume ->
    # CV) rather than the verb the Run button means, and a rebuild would put
    # the noun back from the translation cache, so the fix is claimed as a
    # record rather than hand-edited into the catalog (items 397, 406).
    #
    # 99 -> 116 on 2026-09-15, +17/-0: the magnifier's sixteen captions
    # (407), plus 'Overlap', which read 'Rupture' (a break) on the same card
    # and is now 'Chevauchement'.
    #
    # 116 -> 169 on 2026-09-15, +53/-0, for the 08:15 integration batch:
    #    2  the `segmentation_backend` label and tooltip (404/405),
    #       2026-09-15-segmentation-backend.json
    #   51  corrections from reading every row the batch's GPU pass changed,
    #       2026-09-15-integration-review.json
    # The live count minus the sources in those two files is 116, and they
    # share a source with no other file or with each other.
    #
    # 169 -> 179 on 2026-09-15, +10/-0, for item 286: the same ten
    # performance-level strings as the Swedish note above, in
    # 2026-09-15-performance-levels.json. The live count minus its ten is 169.
    #
    # 179 -> 197 on 2026-09-15, +21/-3, for the magnifier's border option and
    # whole-image mode (407): the same 21 new records and the same 3 retired
    # wordings as Swedish. 179 + 21 - 3 = 197 is the live count, the live
    # count minus the new file's sources is 176 (179 - 3), and the file shares
    # no source with any other.
    # +2 on 2026-09-15, both from 387's selectivity index on the
    # Dose-Response screen: "Host response" and its tooltip, in
    # 2026-09-15-dose-response-host-readout.json. Staged for the catalog
    # pass rather than regenerated here, as the catalog lane asked.
    # +4 more on 2026-09-15, from 387's checkerboard scoring: "Second
    # compound", its tooltip, "Bliss independence" and "Loewe
    # additivity", in 2026-09-15-dose-response-combination.json.
    # 203 -> 238 on 2026-09-15, +35/-0, for runtime pass A: 10 Dose-Response
    # terms, 10 OPS captions, "Controls" and 14 grid captions, in the same
    # files as the Swedish note above. French "Concentration" and "Doses" are
    # MANUAL_UI identity rows in the builder, not records, so neither counts
    # here. 203 + 35 = 238 is the live count, the live count minus the four
    # files' 35 sources is 203, and they share no source with any other file.
    #
    # 238 -> 244 on 2026-09-15, +6/-0: Make Masks' "Load test data…" tooltip
    # and its five status strings (412), 412-make-masks-demo.json. The live
    # count minus that file's six sources is 238, and none of the six is in
    # any other file.
    #
    # 244 -> 259 on 2026-09-15, +15/-0: 416's update dialog and removal
    # reasons, 2026-09-15-update-removes-old-installs.json, the same fifteen
    # as the Swedish note above. The live count minus them is 244.
    #
    # 259 -> 274 on 2026-09-15, +18/-3: the combined 417 settings/model and
    # drag-to-merge records, matching the Swedish sets above. Measured on
    # the combined source: removing the eighteen distinct new sources leaves
    # 256 = 259 - 3, and 256 + 18 = 274. No unrelated pin was moved.
    # 274 -> 277: the same three report sources gain reviewed French wording;
    # no source/key changes or retired records. Subtracting them returns 274.
    #
    # 277 -> 310 on 2026-09-20, for the reason written at length in the
    # Swedish note above: the loader had been raising since 2026-09-15, so
    # neither count could be checked while three commits added records.
    #
    # THE FRENCH ARITHMETIC, in raw records: 280 at the pin, +42 for 418
    # (c0b2c5227), +7 for the settings packs (a64a93e47), = 329; this pass
    # removes 15 whose pinned English no longer exists, leaving 314. The
    # assertion counts DISTINCT sources, and 4 of those raw records repeat a
    # source another record already carries, so 310.
    # +8/-0 on 2026-09-21: current Make Masks readouts. Subtracting
    # this file's distinct sources restores the previously verified count.
    readouts = json.loads((ROOT / "docs/i18n/reviewed/runtime/fr/"
                           "2026-09-21-make-masks-readouts.json").read_text(encoding="utf-8"))
    added_sources = {record["source"] for record in readouts["records"]}
    assert len(added_sources) == 7  # Quality already has a compact owner.
    assert added_sources <= reviewed.keys()
    background = json.loads((ROOT / "docs/i18n/reviewed/runtime/fr/"
                              "2026-09-21-organelle-background.json").read_text(encoding="utf-8"))
    background_sources = {record["source"] for record in background["records"]}
    assert len(background_sources) == 8
    assert not added_sources & background_sources
    assert background_sources <= reviewed.keys()
    samples = json.loads((ROOT / "docs/i18n/reviewed/runtime/fr/"
                           "2026-09-21-dataset-sample-counts.json").read_text(encoding="utf-8"))
    samples = _with_training_sample_replacements(
        samples, "fr", "2026-09-21-dataset-sample-counts.json")
    sample_sources = {record["source"] for record in samples["records"]}
    assert len(sample_sources) == 3
    assert sample_sources <= reviewed.keys()
    assert not sample_sources & (added_sources | background_sources)
    # Three retired chrome/template sources are preserved in the September 23 archive.
    # 305 -> 293 on 2026-09-25, item 511: the twelve fr records of the
    # retired cell, nucleus and pathogen mean bounds were deleted from
    # 2026-09-15-mask-mean-bounds.json (the English is gone).
    # 293 -> 290 on 2026-10-11: three records retired by the N678 factual review.
    # 290 -> 289 on 2026-10-17 (N703): the 2026-09-15 update prompt left the source; its record is archived.
    assert len(reviewed.keys() - added_sources - background_sources - sample_sources) + len(base_retired) == 289
    # +8/-0: four source-bound background labels and four scientific tooltips.
    assert len(reviewed.keys() - background_sources - sample_sources) + len(base_retired) == 296  # 297 -> 296 on 2026-10-17 (N703: a retired per-adapter update prompt was archived).  # 300 -> 297 on 2026-10-11 (N678: three retired). Item 511 retirement (2026-09-25): -12.
    assert len(reviewed.keys() - sample_sources) + len(base_retired) == 304  # 305 -> 304 on 2026-10-17 (N703: a retired per-adapter update prompt was archived).  # Item 511 retirement (2026-09-25): -12.  # 308 -> 305 on 2026-10-11 (N678 retirements)
    assert len(reviewed) + len(base_retired) == 307  # 308 -> 307 on 2026-10-17 (N703: a retired per-adapter update prompt was archived).  # Features, Controls and Quality use compact rows. Item 511 retirement (2026-09-25): -12.  # 311 -> 308 on 2026-10-11 (N678 retirements)
    assert len(older_all_sources - preview_sources - normalized_sources) + len(base_retired) == 314  # 315 -> 314 on 2026-10-17 (N703: a retired per-adapter update prompt was archived).  # Item 511 retirement (2026-09-25): -12.  # 318 -> 315 on 2026-10-11 (N678 retirements)
    assert len(older_all_sources - normalized_sources) + len(base_retired) == 319  # 320 -> 319 on 2026-10-17 (N703: a retired per-adapter update prompt was archived).  # Item 511 retirement (2026-09-25): -12.  # 323 -> 320 on 2026-10-11 (N678 retirements)
    assert len(older_all_sources) + len(base_retired) == 324  # 325 -> 324 on 2026-10-17 (N703: a retired per-adapter update prompt was archived).  # Item 511 retirement (2026-09-25): -12.  # 328 -> 325 on 2026-10-11 (N678 retirements)
    # Item463 retired the one superseded download tooltip, proven above.
    assert len(all_reviewed.keys() - refresh_sources - subsequent_sources - debt_sources - inherited_sources) + len(base_retired) == 334  # 335 -> 334 on 2026-10-17 (N703: a retired per-adapter update prompt was archived).  # 338 -> 335 on 2026-10-11 (N678 retirements)
    # 316 (71071b6c6) retired 17 setup and sign-in captions from the four slices to _ROWS.
    assert len(all_reviewed.keys() - subsequent_sources - debt_sources - inherited_sources) + len(cohort_retired) + len(published_retired) == 603  # 612 -> 611 on 2026-10-17 (N703: a retired per-adapter update prompt was archived).  # 600b (2026-09-29): -1, the Features tooltip retired.  # 619 -> 613 on 2026-10-11 (N678 retirements)  # 611 -> 603 on 2026-10-17 (item 678 Home app-text retirements)
    assert len((all_reviewed.keys() | _superseded_enhancement_sources("fr", all_reviewed)) - debt_sources - inherited_sources) + len(cohort_retired) + len(published_retired) + len(_requested_detection_retirements('fr', '2026-09-22-detection-panel.json')) + len(_later_cohort_retirements('fr', all_reviewed)) == 1654  # 2026-10-24: -1, the old Timelapse workflow-map guidance retired when Timelapse moved under Mask's Time control.  # 1664 -> 1663 on 2026-10-17 (N703: a retired per-adapter update prompt was archived).  # 2026-10-10: -1, the plaque_model tooltip retired when toxoplasma_plaque_v3 became the default.  # 2026-10-04: -2, the withdrawn Plasmodium/Candida guide notes.  # 2026-10-02 (item 43): -2, 603423d0a retired the two later-cohort workflow phrases (778 -> 776 above).  # 2026-10-01: -2, the two FEATURES workflow-map phrases retired.  # 591-597 (2026-09-29): -1, the renamed "Cloud" category caption.  # 600b (2026-09-29): -1, the Features tooltip retired.  # 1671 -> 1665 on 2026-10-11 (N678 retirements)  # 707 (2026-10-16): 1665 -> 1664, Make Masks layout captions retired and added.  # 1663 -> 1655 on 2026-10-17 (item 678 Home app-text retirements)
    assert not published_retired & current_values
    for source, translated in all_reviewed.items():
        assert source in current_values
        assert not _translation_rejection_reasons(
            source,
            translated,
            "fr",
            force=_looks_translatable(source),
        )
