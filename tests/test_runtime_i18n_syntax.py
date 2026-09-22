"""Focused syntax contracts for generated runtime localization catalogs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


def _new_download_sources(language: str, reviewed: dict[str, str]) -> set[str]:
    """Account for the Import and synthetic Invasion review records separately."""
    sources: set[str] = set()
    for filename, expected in (("import-examples", 9), ("synthetic-invasion", 2)):
        document = json.loads((ROOT / "docs/i18n/reviewed/runtime" / language /
                               f"2026-09-21-{filename}.json").read_text())
        added = {record["source"] for record in document["records"]}
        assert len(document["records"]) == len(added) == expected
        assert added <= reviewed.keys()
        assert not added & sources
        sources.update(added)
    assert len(sources) == 11
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
        ("2026-09-22-workflow-pathways.json", 47),
        ("2026-09-22-figure-settings.json", 41),
        ("2026-09-22-home-sample-project.json", 6),
        ("2026-09-22-inversion-controls.json", 2 if language == "sv" else 1),
    ):
        document = json.loads((ROOT / "docs/i18n/reviewed/runtime" / language /
                               filename).read_text())
        added = {record["source"] for record in document["records"]}
        assert len(document["records"]) == len(added) == expected
        assert added <= reviewed.keys()
        assert not added & sources
        sources.update(added)
    assert len(sources) == (150 if language == "sv" else 149)
    overlay = json.loads((ROOT / "docs/i18n/reviewed/runtime" / language /
                          "2026-09-22-plaque-overlays.json").read_text())
    overlay_sources = {record["source"] for record in overlay["records"]}
    # Plaque model has both a UI caption and a setting-label record.
    assert len(overlay["records"]) == 12
    assert len(overlay_sources) == 11
    assert overlay_sources <= reviewed.keys()
    assert not overlay_sources & sources
    sources.update(overlay_sources)
    assert len(sources) == (161 if language == "sv" else 160)
    panel = json.loads((ROOT / "docs/i18n/reviewed/runtime" / language /
                        "2026-09-22-detection-panel.json").read_text())
    panel_sources = {record["source"] for record in panel["records"]}
    assert len(panel["records"]) == len(panel_sources) == 6
    assert panel_sources <= reviewed.keys()
    assert not panel_sources & sources
    sources.update(panel_sources)
    assert len(sources) == (167 if language == "sv" else 166)
    for filename, expected in (("form-labels-a", 78), ("sign-in-status", 1)):
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
    assert len(sources) == (245 if language == "sv" else 244)
    return sources


def _compact_tooltip_sources(language: str) -> set[str]:
    """Two concise replacements retain the scientific review cohort's size."""
    path = ROOT / "docs/i18n/reviewed/runtime" / language / "2026-09-22-compact-tooltips.json"
    records = json.loads(path.read_text())["records"]
    assert len(records) == 2
    assert {record["table"] for record in records} == {"setting_tooltips"}
    assert {record["key"] for record in records} == {"annotation_source", "metadata_type"}
    return {record["source"] for record in records}


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
    ui_sources = {record["source"] for record in ui_refresh["records"]}
    assert len(ui_refresh["records"]) == 266  # Channels already has a compact owner.
    assert len(ui_sources) == 263
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
    older_all_sources = all_reviewed.keys() - download_sources - subsequent_sources
    reviewed = {source: value for source, value in all_reviewed.items()
                if source not in ui_sources | example_sources | preview_sources | normalized_sources | download_sources | subsequent_sources}
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
    assert len(older_sources - added_sources - background_sources) == 317
    # +8/-0: four source-bound background labels and four scientific tooltips.
    assert len(older_sources - background_sources) == 324
    assert len(older_sources) == 332
    assert len(reviewed.keys() - sample_sources) == 371  # +39 scientific sources.
    assert len(reviewed) == 374  # Features, Controls and Quality use compact rows.
    # The new panel cohort also reuses the earlier whole-field model tooltip.
    assert len(older_all_sources - example_sources - preview_sources - normalized_sources) == 636
    assert len(older_all_sources - preview_sources - normalized_sources) == 643
    assert len(older_all_sources - normalized_sources) == 648
    assert len(older_all_sources) == 653
    assert len(all_reviewed.keys() - subsequent_sources) == 664
    assert len(all_reviewed) == 909  # 77 new form sources, one sign-in, four transfers.
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
    assert len(refresh["records"]) == len(refresh_sources) == 77
    assert not refresh_sources & _compact_tooltip_sources("fr")
    refresh_sources |= _compact_tooltip_sources("fr")
    assert len(refresh_sources) == 79
    assert refresh_sources <= all_reviewed.keys()
    second = json.loads((ROOT / "docs/i18n/reviewed/runtime/fr/"
                         "2026-09-21-runtime-second-slice.json").read_text())
    second_sources = {record["source"] for record in second["records"]}
    assert len(second["records"]) == len(second_sources) == 74
    assert second_sources <= all_reviewed.keys()
    assert not refresh_sources & second_sources
    refresh_sources |= second_sources
    # Two third-slice and three fourth-slice captions left with the old panel.
    for filename, expected in (("third", 73), ("fourth", 76)):
        document = json.loads((ROOT / "docs/i18n/reviewed/runtime/fr/"
                               f"2026-09-21-runtime-{filename}-slice.json").read_text())
        added = {record["source"] for record in document["records"]}
        assert len(document["records"]) == len(added) == expected
        assert added <= all_reviewed.keys()
        assert not added & refresh_sources
        refresh_sources |= added
    assert len(refresh_sources) == 302
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
    older_all_sources = all_reviewed.keys() - download_sources - refresh_sources - subsequent_sources
    reviewed = {source: value for source, value in all_reviewed.items()
                if source not in example_sources | preview_sources | normalized_sources | download_sources | refresh_sources | subsequent_sources}
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
    sample_sources = {record["source"] for record in samples["records"]}
    assert len(sample_sources) == 3
    assert sample_sources <= reviewed.keys()
    assert not sample_sources & (added_sources | background_sources)
    assert len(reviewed.keys() - added_sources - background_sources - sample_sources) == 308
    # +8/-0: four source-bound background labels and four scientific tooltips.
    assert len(reviewed.keys() - background_sources - sample_sources) == 315
    assert len(reviewed.keys() - sample_sources) == 323
    assert len(reviewed) == 326  # Features, Controls and Quality use compact rows.
    assert len(older_all_sources - preview_sources - normalized_sources) == 333
    assert len(older_all_sources - normalized_sources) == 338
    assert len(older_all_sources) == 343
    assert len(all_reviewed.keys() - refresh_sources - subsequent_sources) == 354
    assert len(all_reviewed.keys() - subsequent_sources) == 660
    assert len(all_reviewed) == 904  # 77 new form sources, one sign-in, four transfers.
    for source, translated in all_reviewed.items():
        assert source in current_values
        assert not _translation_rejection_reasons(
            source,
            translated,
            "fr",
            force=_looks_translatable(source),
        )
