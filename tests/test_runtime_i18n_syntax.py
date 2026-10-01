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
                               f"2026-09-21-{filename}.json").read_text())
        added = {record["source"] for record in document["records"]}
        assert len(document["records"]) == len(added) == expected
        assert added <= reviewed.keys()
        assert not added & sources
        sources.update(added)
    # Item463 replaced the unsegmented-example tooltip. Preserve the original
    # two-record evidence and prove that only the superseded tooltip retired.
    archive = json.loads((ROOT / "features/data/463_retired_runtime_review_2026-09-27" /
                          f"{language}.json").read_text())["records"]
    old_invasion = {record["source"] for record in archive}
    assert len(archive) == len(old_invasion) == 2
    assert len(old_invasion - sources) == 1
    assert not (old_invasion - sources) & reviewed.keys()
    assert len(sources) == 10
    return sources


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
                               filename).read_text())
        added = {record["source"] for record in document["records"]}
        assert len(document["records"]) == len(added) == expected
        assert added <= reviewed.keys()
        assert not added & sources
        sources.update(added)
    assert len(sources) == (142 if language == "sv" else 141)
    overlay = json.loads((ROOT / "docs/i18n/reviewed/runtime" / language /
                          "2026-09-22-plaque-overlays.json").read_text())
    overlay_sources = {record["source"] for record in overlay["records"]}
    # Plaque model has both a UI caption and a setting-label record.
    assert len(overlay["records"]) == 12
    assert len(overlay_sources) == 11
    assert overlay_sources <= reviewed.keys()
    assert not overlay_sources & sources
    sources.update(overlay_sources)
    assert len(sources) == (153 if language == "sv" else 152)
    panel = json.loads((ROOT / "docs/i18n/reviewed/runtime" / language /
                        "2026-09-22-detection-panel.json").read_text())
    panel_sources = {record["source"] for record in panel["records"]}
    assert len(panel["records"]) == len(panel_sources) == 6
    assert panel_sources <= reviewed.keys()
    assert not panel_sources & sources
    sources.update(panel_sources)
    assert len(sources) == (159 if language == "sv" else 158)
    # Instruction 316 retired the one sign-in-status record to _ROWS.
    for filename, expected in (("form-labels-a", 77), ("sign-in-status", 0),
                               ("enhancement-and-scale", 9),
                               ("organism-identities", 2),
                               ("threshold-and-histogram", 13)):
        document = json.loads((ROOT / "docs/i18n/reviewed/runtime" / language /
                               f"2026-09-22-{filename}.json").read_text())
        added = {record["source"] for record in document["records"]}
        assert len(document["records"]) == len(added) == expected
        assert added <= reviewed.keys()
        if filename == "form-labels-a":
            # The UI row reuses an already reviewed setting-label source.
            assert "Crop size" in added
            added.remove("Crop size")
        assert not added & sources
        sources.update(added)
    assert len(sources) == (259 if language == "sv" else 258)
    report = json.loads((ROOT / "tests/data/release_contracts/411_runtime_review_cohorts_2026-09-23.json").read_text())["languages"][language]
    folder = ROOT / "docs/i18n/reviewed/runtime" / language
    later_sources: set[str] = set()
    later_names = {row["name"] for row in report["files"]}
    for row in report["files"]:
        path = folder / row["name"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        records = json.loads(path.read_text())["records"]
        values = {record["source"] for record in records}
        assert len(records) == row["records"] and len(values) == row["sources"]
        assert values <= reviewed.keys()
        later_sources.update(values)
    earlier_sources = {
        record["source"] for path in folder.glob("*.json") if path.name not in later_names
        for record in json.loads(path.read_text())["records"]
    }
    additions = later_sources - earlier_sources
    # 781 -> 780 on 2026-09-25: nightly merge 5c03e8bda removed one retired PSF
    # help record from 2026-09-23-psf-help.json (11 -> 10 records).
    assert len(additions) == report["later_distinct_additions"] == 780
    assert hashlib.sha256(json.dumps(sorted(additions), ensure_ascii=False).encode()).hexdigest() == report["added_sources_sha256"]
    assert not sources & additions
    for filename, record_count, source_count in (
            # 3 -> 2 records on 2026-09-29: items 591-597 renamed the "Cloud"
            # category, so its caption record was deleted (key gone).
            ("2026-09-27-mask-cloud-category.json", 2, 1),
            ("2026-09-28-runtime-577-585.json", 20, 20)):
        document = json.loads((folder / filename).read_text())
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
    records = json.loads(path.read_text())["records"]
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


def _with_training_sample_replacements(document, language, filename):
    archive = ROOT / "features/data/450_451_retired_runtime_review_2026-09-27" / language / filename
    if not archive.exists():
        return document
    original = json.loads(archive.read_text())["records"]
    retained = document["records"]
    # 600b, 2026-09-29: Make Masks' Features button was removed, so its
    # tooltip's reviewed record was deleted (key gone); it is not one of the
    # training-sample replacements below.
    features = {row for row in (json.dumps(r, sort_keys=True) for r in original)
                if json.loads(row)["source"] in _RETIRED_BY_600B}
    features = [json.loads(row) for row in features]
    original = [row for row in original if row not in features]
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
                              "2026-09-27-training-samples.json").read_text())["records"]
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
                               / name).read_text())
        sources |= {record["source"] for record in document["records"]}
    # 31 -> 30 on 2026-09-29: the "Measurement Backend (Alpha)" caption was
    # renamed to its "α" form (591-597); the old record was deleted.
    assert len(sources) == 30
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
             for record in json.loads(path.read_text())["records"]]
    # The second pass covers sources added on nightly after the first pass
    # was prepared (items 509-530); it is optional here so a locale can land
    # its first pass alone, and it must not overlap the first.
    second = [record for path in paths if "-second-pass-" in path.name
              for record in json.loads(path.read_text())["records"]]
    sources = {record["source"] for record in first}
    assert len(first) == len(sources) == expected
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
             for record in json.loads(path.read_text())["records"]]
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
              for record in json.loads(path.read_text())["records"]]
    newest = {record["source"] for record in fourth}
    assert len(fourth) == len(newest) and not newest & sources
    sources |= newest
    # The fifth pass (2026-09-26) is one more slice: items 536, 550, 577 and
    # the profiling and cell-cycle settings that reached nightly after the
    # fourth. Every source was pending, so it overlaps no earlier pass.
    fifth = [record for path in sorted(folder.glob("2026-09-26-runtime-debt-fifth-pass-*.json"))
             for record in json.loads(path.read_text())["records"]]
    latest5 = {record["source"] for record in fifth}
    assert len(fifth) == len(latest5) and not latest5 & sources
    sources |= latest5
    # The sixth pass (2026-09-27) is one more slice: items 539, 571, 579,
    # 582 and the other captions nightly added after the fifth.
    sixth = [record for path in sorted(folder.glob("2026-09-27-runtime-debt-sixth-pass-*.json"))
             for record in json.loads(path.read_text())["records"]]
    latest6 = {record["source"] for record in sixth}
    assert len(sixth) == len(latest6) and not latest6 & sources
    sources |= latest6
    # Direct Codex-reviewed delta: 129 new sources plus11 extraction repairs,
    # minus the Spotiflow identity; Hindi also resolves35 historical fallbacks.
    seventh = json.loads((folder / "2026-09-27-runtime-codex-delta.json").read_text())["records"]
    latest7 = {record["source"] for record in seventh}
    # 139 -> 136 (hi 174 -> 171) on 2026-09-29: items 591-597 renamed the
    # Event Detection, GPU Measurement and Segmentation Robustness categories
    # to their "α" captions, so those three records were deleted (key gone).
    assert len(seventh) == len(latest7) == (171 if language == "hi" else 136)
    assert not latest7 & sources
    sources |= latest7
    discovery = json.loads((folder / "2026-09-27-gpu-discovery.json").read_text())["records"]
    discovery_sources = {record["source"] for record in discovery}
    assert len(discovery) == 1
    assert discovery_sources == {"Checking compatible GPUs…"}
    assert not discovery_sources & sources
    sources |= discovery_sources
    # The 2026-09-29 seventh runtime pass: the renamed "α" categories, the
    # channel-sort and consolidate dialogs, Noise2Void and the new category
    # help (items 591-597), one more slice that overlaps nothing earlier.
    pass7 = [record for path in sorted(folder.glob("2026-09-29-runtime-debt-seventh-pass-*.json"))
             for record in json.loads(path.read_text())["records"]]
    latest_p7 = {record["source"] for record in pass7}
    assert len(pass7) == len(latest_p7) and not latest_p7 & sources
    sources |= latest_p7
    # 2026-09-29 (474): the eight organism tile usage notes ("Opens ..."),
    # which entered the runtime inventory only now; another disjoint slice.
    notes = json.loads((folder / "2026-09-29-organism-workflow-notes.json")
                       .read_text())["records"]
    note_sources = {record["source"] for record in notes}
    assert len(notes) == len(note_sources) == 8 and not note_sources & sources
    sources |= note_sources
    # The 2026-09-30 eighth runtime pass: the Organize for Measure popup,
    # its regex teaching and drop-classification dialogs and the popup
    # table's view options (items 600 and 592); another disjoint slice.
    pass8 = [record for path in sorted(folder.glob("2026-09-30-runtime-debt-eighth-pass-*.json"))
             for record in json.loads(path.read_text())["records"]]
    latest_p8 = {record["source"] for record in pass8}
    assert len(pass8) == len(latest_p8) and not latest_p8 & sources
    sources |= latest_p8
    # The 2026-09-30 ninth runtime pass: the colony_detector setting label
    # and tooltip (item 542); another disjoint slice.
    pass9 = [record for path in sorted(folder.glob("2026-09-30-runtime-debt-ninth-pass-*.json"))
             for record in json.loads(path.read_text())["records"]]
    latest_p9 = {record["source"] for record in pass9}
    assert len(pass9) == len(latest_p9) and not latest_p9 & sources
    sources |= latest_p9
    # The 2026-09-30 tenth runtime pass: item 598's contribute-dialog
    # captions (the third-pass record of the old linked thank-you was retired
    # in place) and the Make Masks button re-layout (Upload data, the Load
    # test data menu, the Uncertainty setting).
    pass10 = [record for path in sorted(folder.glob("2026-09-30-runtime-debt-tenth-pass-*.json"))
              for record in json.loads(path.read_text())["records"]]
    latest_p10 = {record["source"] for record in pass10}
    assert len(pass10) == len(latest_p10) and not latest_p10 & sources
    sources |= latest_p10
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
    # +269 distinct UI/category sources, with three OPS descriptions shared
    # between both tables (272 records). The detection-panel consolidation
    # retired five source captions, preserving their records in the archive.
    ui_refresh = json.loads((ROOT / "docs/i18n/reviewed/runtime/sv/"
                              "2026-09-21-runtime-ui-refresh.json").read_text())
    ui_refresh = _with_training_sample_replacements(
        ui_refresh, "sv", "2026-09-21-runtime-ui-refresh.json")
    ui_sources = {record["source"] for record in ui_refresh["records"]}
    # Item 511 retired four Make Masks filter captions (the fixed bounds'
    # button, ledger, card and placeholder help): 263 -> 259, 260 -> 256.
    # Instruction 316 then retired the 17 setup and sign-in captions that
    # moved to _ROWS (kept under retired_records): 259 -> 242, 256 -> 239.
    # 600b, 2026-09-29: the Features button's tooltip was retired (Make
    # Masks' button removed): 242 -> 241, 239 -> 238.
    assert len(ui_refresh["records"]) == 241  # Three old threshold/histogram reviews archived.
    assert len(ui_sources) == 238
    assert ui_sources <= reviewed.keys()
    all_reviewed = reviewed
    examples = json.loads((ROOT / "docs/i18n/reviewed/runtime/sv/"
                            "2026-09-21-assay-and-plate-examples.json").read_text())
    example_sources = {record["source"] for record in examples["records"]}
    assert len(example_sources) == 7
    assert example_sources <= all_reviewed.keys()
    assert not example_sources & ui_sources
    previews = json.loads((ROOT / "docs/i18n/reviewed/runtime/sv/"
                            "2026-09-21-preview-refresh.json").read_text())
    preview_sources = {record["source"] for record in previews["records"]}
    assert len(preview_sources) == 5
    assert preview_sources <= all_reviewed.keys()
    assert not preview_sources & example_sources
    normalized = json.loads((ROOT / "docs/i18n/reviewed/runtime/sv/"
                              "2026-09-21-normalized-detection.json").read_text())
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
    debt_sources = _runtime_debt_sources("sv", all_reviewed, 709)  # 710 -> 709 on 2026-09-29: the "Point Spread Function" category caption was renamed (591-597)
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
                           "2026-09-21-make-masks-readouts.json").read_text())
    added_sources = {record["source"] for record in readouts["records"]}
    assert len(added_sources) == 7  # Quality already has a compact owner.
    assert added_sources <= reviewed.keys()
    background = json.loads((ROOT / "docs/i18n/reviewed/runtime/sv/"
                              "2026-09-21-organelle-background.json").read_text())
    background_sources = {record["source"] for record in background["records"]}
    assert len(background_sources) == 8
    assert not added_sources & background_sources
    assert background_sources <= reviewed.keys()
    samples = json.loads((ROOT / "docs/i18n/reviewed/runtime/sv/"
                           "2026-09-21-dataset-sample-counts.json").read_text())
    samples = _with_training_sample_replacements(
        samples, "sv", "2026-09-21-dataset-sample-counts.json")
    sample_sources = {record["source"] for record in samples["records"]}
    assert len(sample_sources) == 3
    assert sample_sources <= reviewed.keys()
    assert not sample_sources & (added_sources | background_sources)
    scientific = json.loads((ROOT / "docs/i18n/reviewed/runtime/sv/"
                              "2026-09-21-scientific-settings.json").read_text())
    scientific_sources = {record["source"] for record in scientific["records"]}
    assert len(scientific_sources) == 37
    assert not scientific_sources & _compact_tooltip_sources("sv")
    scientific_sources |= _compact_tooltip_sources("sv")
    assert len(scientific_sources) == 39
    assert scientific_sources <= reviewed.keys()
    assert not scientific_sources & (added_sources | background_sources)
    assert not sample_sources & scientific_sources
    older_sources = reviewed.keys() - scientific_sources - sample_sources
    # Four retired chrome/template sources are preserved in the September 23 archive.
    # 313 -> 306 on 2026-09-25, item 511: the maintainer retired the cell,
    # nucleus and pathogen mean-bound settings into object_filters rows, and
    # the seven sv records of their labels and tooltips were deleted from
    # 2026-09-15-mask-mean-bounds.json (the English is gone).
    assert len(older_sources - added_sources - background_sources) == 305  # 316 fourth pass (2026-09-26): -1, the percentiles tooltip record left 2026-08-14-exact-final.json (its English changed).
    # +8/-0: four source-bound background labels and four scientific tooltips.
    assert len(older_sources - background_sources) == 312  # Item 511 retirement (2026-09-25): -7.  # 316 fourth pass (2026-09-26): -1, the percentiles tooltip record left 2026-08-14-exact-final.json (its English changed).
    assert len(older_sources) == 320  # Item 511 retirement (2026-09-25): -7.  # 316 fourth pass (2026-09-26): -1, the percentiles tooltip record left 2026-08-14-exact-final.json (its English changed).
    assert len(reviewed.keys() - sample_sources) == 359  # +39 scientific sources. Item 511 retirement (2026-09-25): -7.  # 316 fourth pass (2026-09-26): -1, the percentiles tooltip record left 2026-08-14-exact-final.json (its English changed).
    assert len(reviewed) == 362  # Features, Controls and Quality use compact rows. Item 511 retirement (2026-09-25): -7.  # 316 fourth pass (2026-09-26): -1, the percentiles tooltip record left 2026-08-14-exact-final.json (its English changed).
    # The new panel cohort also reuses the earlier whole-field model tooltip.
    # 316 (71071b6c6) retired 17 setup and sign-in captions to _ROWS: -17 below;
    # the total also loses its sign-in-status record and a superseded psf-help record.
    assert len(older_all_sources - example_sources - preview_sources - normalized_sources) == 599  # 600b (2026-09-29): -1, the Features tooltip retired.  # Item 511 retired four filter captions. Item 511 retirement (2026-09-25): -7.  # 316 fourth pass (2026-09-26): -1, the percentiles tooltip record left 2026-08-14-exact-final.json (its English changed).
    assert len(older_all_sources - preview_sources - normalized_sources) == 606  # 600b (2026-09-29): -1, the Features tooltip retired.  # Item 511 retired four filter captions. Item 511 retirement (2026-09-25): -7.  # 316 fourth pass (2026-09-26): -1, the percentiles tooltip record left 2026-08-14-exact-final.json (its English changed).
    assert len(older_all_sources - normalized_sources) == 611  # 600b (2026-09-29): -1, the Features tooltip retired.  # Item 511 retirement (2026-09-25): -7.  # 316 fourth pass (2026-09-26): -1, the percentiles tooltip record left 2026-08-14-exact-final.json (its English changed).  # 600b (2026-09-29): -1, the Features tooltip retired.
    assert len(older_all_sources) == 616  # 600b (2026-09-29): -1, the Features tooltip retired.  # Item 511 retirement (2026-09-25): -7.  # 316 fourth pass (2026-09-26): -1, the percentiles tooltip record left 2026-08-14-exact-final.json (its English changed).  # 600b (2026-09-29): -1, the Features tooltip retired.
    # Item463 retired one superseded download tooltip; its full old evidence
    # and exact set difference are checked by _new_download_sources above.
    assert len(all_reviewed.keys() - subsequent_sources - debt_sources - inherited_sources) == 626  # 600b (2026-09-29): -1, the Features tooltip retired.
    assert len(all_reviewed.keys() - debt_sources - inherited_sources) == 1686  # 591-597 (2026-09-29): -1, the renamed "Cloud" category caption.  # 600b (2026-09-29): -1, the Features tooltip retired.
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
    refresh = json.loads((ROOT / "docs/i18n/reviewed/runtime/fr/"
                          "2026-09-21-runtime-first-slice.json").read_text())
    refresh_sources = {record["source"] for record in refresh["records"]}
    # Item 511 retired one filter caption from each of the four slices, and
    # instruction 316 retired the setup and sign-in captions that moved to
    # _ROWS (2, 7, 4 and 4 per slice, kept under retired_records).
    assert len(refresh["records"]) == len(refresh_sources) == 73
    assert not refresh_sources & _compact_tooltip_sources("fr")
    refresh_sources |= _compact_tooltip_sources("fr")
    assert len(refresh_sources) == 75
    assert refresh_sources <= all_reviewed.keys()
    second = json.loads((ROOT / "docs/i18n/reviewed/runtime/fr/"
                         "2026-09-21-runtime-second-slice.json").read_text())
    second = _with_training_sample_replacements(
        second, "fr", "2026-09-21-runtime-second-slice.json")
    second_sources = {record["source"] for record in second["records"]}
    # 64 -> 63 on 2026-09-29: 600b retired the Features button's tooltip.
    assert len(second["records"]) == len(second_sources) == 63
    assert second_sources <= all_reviewed.keys()
    assert not refresh_sources & second_sources
    refresh_sources |= second_sources
    # Two third-slice and three fourth-slice captions left with the old panel.
    for filename, expected in (("third", 68), ("fourth", 71)):
        document = json.loads((ROOT / "docs/i18n/reviewed/runtime/fr/"
                               f"2026-09-21-runtime-{filename}-slice.json").read_text())
        document = _with_training_sample_replacements(
            document, "fr", f"2026-09-21-runtime-{filename}-slice.json")
        added = {record["source"] for record in document["records"]}
        assert len(document["records"]) == len(added) == expected
        assert added <= all_reviewed.keys()
        assert not added & refresh_sources
        refresh_sources |= added
    assert len(refresh_sources) == 277  # 600b (2026-09-29): -1, the Features tooltip retired.  # Three superseded threshold/histogram sources.
    actions = json.loads((ROOT / "docs/i18n/reviewed/runtime/fr/"
                          "2026-09-21-action-labels.json").read_text())
    action_sources = {record["source"] for record in actions["records"]}
    assert len(actions["records"]) == len(action_sources) == 5
    assert action_sources <= all_reviewed.keys()
    assert not action_sources & refresh_sources
    refresh_sources |= action_sources
    examples = json.loads((ROOT / "docs/i18n/reviewed/runtime/fr/"
                            "2026-09-21-assay-and-plate-examples.json").read_text())
    example_sources = {record["source"] for record in examples["records"]}
    assert len(example_sources) == 7
    assert example_sources <= all_reviewed.keys()
    previews = json.loads((ROOT / "docs/i18n/reviewed/runtime/fr/"
                            "2026-09-21-preview-refresh.json").read_text())
    preview_sources = {record["source"] for record in previews["records"]}
    assert len(preview_sources) == 5
    assert preview_sources <= all_reviewed.keys()
    assert not preview_sources & example_sources
    normalized = json.loads((ROOT / "docs/i18n/reviewed/runtime/fr/"
                              "2026-09-21-normalized-detection.json").read_text())
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
    debt_sources = _runtime_debt_sources("fr", all_reviewed, 709)  # 710 -> 709 on 2026-09-29: the "Point Spread Function" category caption was renamed (591-597)
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
                           "2026-09-21-make-masks-readouts.json").read_text())
    added_sources = {record["source"] for record in readouts["records"]}
    assert len(added_sources) == 7  # Quality already has a compact owner.
    assert added_sources <= reviewed.keys()
    background = json.loads((ROOT / "docs/i18n/reviewed/runtime/fr/"
                              "2026-09-21-organelle-background.json").read_text())
    background_sources = {record["source"] for record in background["records"]}
    assert len(background_sources) == 8
    assert not added_sources & background_sources
    assert background_sources <= reviewed.keys()
    samples = json.loads((ROOT / "docs/i18n/reviewed/runtime/fr/"
                           "2026-09-21-dataset-sample-counts.json").read_text())
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
    assert len(reviewed.keys() - added_sources - background_sources - sample_sources) == 293
    # +8/-0: four source-bound background labels and four scientific tooltips.
    assert len(reviewed.keys() - background_sources - sample_sources) == 300  # Item 511 retirement (2026-09-25): -12.
    assert len(reviewed.keys() - sample_sources) == 308  # Item 511 retirement (2026-09-25): -12.
    assert len(reviewed) == 311  # Features, Controls and Quality use compact rows. Item 511 retirement (2026-09-25): -12.
    assert len(older_all_sources - preview_sources - normalized_sources) == 318  # Item 511 retirement (2026-09-25): -12.
    assert len(older_all_sources - normalized_sources) == 323  # Item 511 retirement (2026-09-25): -12.
    assert len(older_all_sources) == 328  # Item 511 retirement (2026-09-25): -12.
    # Item463 retired the one superseded download tooltip, proven above.
    assert len(all_reviewed.keys() - refresh_sources - subsequent_sources - debt_sources - inherited_sources) == 338
    # 316 (71071b6c6) retired 17 setup and sign-in captions from the four slices to _ROWS.
    assert len(all_reviewed.keys() - subsequent_sources - debt_sources - inherited_sources) == 619  # 600b (2026-09-29): -1, the Features tooltip retired.
    assert len(all_reviewed.keys() - debt_sources - inherited_sources) == 1678  # 591-597 (2026-09-29): -1, the renamed "Cloud" category caption.  # 600b (2026-09-29): -1, the Features tooltip retired.
    for source, translated in all_reviewed.items():
        assert source in current_values
        assert not _translation_rejection_reasons(
            source,
            translated,
            "fr",
            force=_looks_translatable(source),
        )
