"""Ratcheted coverage for spaCR's two runtime-translation layers.

The compact layer owns captions assembled from registries and first-run data:
there is no literal Qt call for the catalog generator to find.  The external
layer owns the much larger static/widget, setting, category and module-summary
surface and binds every translation to a hash of its English source.  These
tests keep the two contracts complementary instead of copying thousands of
generated records into ``i18n._ROWS``.

Deliberate exclusions from the compact surface are language names written in
their own language, provider/product identities, user-entered values and the
generated setting/static-Qt prose.  Native names and product identities are
not translated; generated prose is independently guarded below by the
external source-hash registry.
"""
from __future__ import annotations

import ast
import hashlib
import re
import sys
from collections import Counter
from importlib import import_module
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Post-sweep compact surface on 2026-08-29.  The count catches additions; the
# digest also catches a replacement that happens to keep the count unchanged.
#
# 205 on 2026-08-30, +4/-1.  Admitted: "Field browser" and "Quarantine or
# restore this field", the category and label for the Q key that quarantines a
# field -- it was bound in the QC field browser and on no map, so the cheat
# sheet told the reader it did not exist.  Also "the QC field browser" and
# "the Annotate and Make Masks screens and the QC field browser", the scopes
# that say where Q and the arrows work.  Retired: "the Annotate and Make Masks
# screens", which the longer scope replaces -- the arrows drive the field
# browser too, and the shorter wording had stopped being true.
# 206 on 2026-09-04, +1/-0.  Admitted: "Import Images", the Import module's own
# caption, which is distinct from the bare "Import" band name already in
# _ROWS.  Each locale reuses its established "Import" verb so the two read as
# one family on the same screen.
# 206 -> 207 on 2026-09-07, +1/-0: "OPS", the fold label on Align. It is
# identical in all nine locales, which IS its translation -- optical pooled
# screening is named by its acronym in the literature these locales publish
# in, and there is no local expansion the way French has ACP for PCA.
# Still 207 on 2026-09-08, +0/-0: no caption was added or removed, one was
# REWORDED. Preferences moved from Ctrl+comma to Ctrl+P after both it and
# Ctrl+H were reported firing nothing -- the menu action and
# `shortcuts.install` each held the key, and Qt answers two holders by
# firing neither -- so the shortcuts help line now names Ctrl+P. The
# substitution is the key's NAME, made identically in all nine locales,
# because a key name is not prose: "Ctrl+P" is what the user's keyboard
# says in every language.
# 207 -> 208 on 2026-09-09: 378's cheat-sheet entry for the Z + scroll
# gesture, "Resize the interface text". It is a SHORTCUT label, so it
# lives in the compact layer with its nine hand-written rows rather than
# in the generated one. 378 deferred it for exactly this reason -- every
# spec's label is a row here -- and named the condition it was waiting
# on: "whenever the catalogs are next rebuilt".
# 208 -> 209 on 2026-09-12, +1/-0. Admitted: "Embeddings", the name 386's
# new module registers in APPS, in the Data section. An app name is compact
# copy by this file's own rule, and this one arrived with its nine rows
# already written -- the discovered set minus `_ROWS` is empty, so the
# literal-row requirement was already satisfied and only the number was
# behind. Measured, not guessed: this set was recomputed at 76921939b, the
# commit that last moved it, and at the merge, and "Embeddings" is the
# entire difference. Nothing was retired and nothing was reworded, which is
# why the count and the digest move together this time rather than the
# digest alone.
# 209 -> 208 on 2026-09-15, +0/-1, for item 286. Retired: "spaCR mode", the
# caption first-run setup gave the performance selector -- the name of the
# control 286 removed. Setup now captions it "Performance", as Preferences
# does, and "Performance" was already on this surface, so nothing arrived.
# PROVED BY SUBTRACTION with this test's own formula: today's set plus
# "spaCR mode" gives e0b2c63f3e43a544..., the previous pin byte for byte.
# Its `_ROWS` row stays; this test only requires a row per caption, not the
# converse.
COMPACT_CAPTION_COUNT = 208
COMPACT_CAPTION_SHA256 = (
    "4663f4f0872bf921d343da2b9909a56e6b8697ccb832936b011b144566b64eb2"
)

# The complementary source-bound layer is pinned separately.  Keys are
# catalog record identities (table plus key), not translated values: adding a
# static caption, setting, category, or module summary must therefore move
# this reviewed inventory deliberately even when the builder can generate a
# source hash automatically.
# MOVED 2026-09-04, and only after the catalogs were verified green: this
# file's own rule is that a digest nobody has seen pass is decoration.
#
# WHAT UNBLOCKED IT is 056e5f1ef, the maintainer's own pass on 2026-09-02 over
# the captions 316's addendum was holding this pin for.  That addendum forbade
# moving these numbers while the strings were still untranslated, and the
# prohibition is spent, not waived -- the strings it named are translated.
#
# WHAT THAT PASS DID NOT COVER, so nobody reads a green digest as more than it
# is: it covered Icelandic, Swedish and German, the three locales the
# maintainer reads.  The other six are still machine-drafted and technically
# reviewed.  The README says exactly that, and the coverage is recorded in
# docs/i18n/REVIEW_SCOPE_2026-09-04.md.
#
# The counts follow the catalogs, never the reverse.  UI and MODULE_SUMMARIES
# are the manifest catching up with canonical; SETTING_LABELS and
# SETTING_TOOLTIPS each gain the one row the settings work added.
# MOVED 2026-09-07, and every one of these is the OPS and suggestion work
# arriving in the manifest rather than anything drifting. SETTING_LABELS
# 1,019 -> 1,077 and SETTING_TOOLTIPS 1,014 -> 1,072 are the stitch, mosaic,
# outline and feature-cache settings; UI 2,817 -> 2,838 is the Suggest button
# and its notices; MODULE_SUMMARIES 66 -> 67 is the OPS module itself. The
# nine locales were regenerated against this manifest in 6aa5a73aa before
# these numbers were touched, which is the order this file's own rule asks
# for: catalogs first, ratchet second, never the reverse.
#
# UI IS 2,838, CORRECTED 2026-09-08 FROM 2,837. The note here used to say
# 2,837 was "one fewer than the regeneration reported and is not a
# discrepancy", the reasoning being that giving "OPS" an exact `_ROWS` row
# moved it out of the generated layer, and the two layers are disjoint by
# contract.
#
# The reasoning is right and the arithmetic double-counted it. "OPS" was
# ALREADY absent from `UI_SOURCES` in 6aa5a73aa, the very commit whose
# regeneration produced 2,838 -- so that number was measured after the move,
# not before it, and subtracting one for the move charged it twice. The
# effect was a ratchet that no build could ever satisfy.
#
# The disjointness itself still holds and is still worth stating: a caption
# belongs to the reviewed compact layer or the generated one, never both.
#
# MOVED DOWN 2026-09-11, and this is the first time these counts have gone
# DOWN. 1,077 -> 1,073 and 1,072 -> 1,068, net -4 across 15 removals and 11
# additions, every one of them 364's settings audit or 388's new pair.
# Reviewed one at a time, as this file's own message demands, by computing
# the canonical identities at 69fd34e44 -- the commit that last moved these
# numbers -- and at the merge, and diffing the sets:
#
#   RENAMED, so one removal and one addition each and no net change:
#     negative_control     -> negative_control_id
#     positive_control     -> positive_control_id
#     controls             -> nontargeting_control_grnas
#     min_cell_count       -> min_cells_per_well
#     min_n                -> min_observations_per_hit
#     expected_end         -> window_length
#     control_wells        -> analysis_excluded_wells, stain_baseline_wells
#                             (one setting that meant two things, split)
#
#   ADDED: bystander_measurements, bystander_reach_in_diameters (388),
#     and ops_gpu.
#
#   REMOVED OUTRIGHT, eight settings 364 judged redundant: denoise,
#     load_path_regex, mask_array, normalization, normalization_scope,
#     normalize_plots, save_to_db, visualize.
#
# CATEGORY_HELP 201 -> 200 is one record and it is the same event: the help
# text that belonged to `save_to_db`, which no longer exists.
#
# NOT A MERGE ARTEFACT, which is what this failure had been recorded as.
# `canonical_sources()` returns 1,073/1,068/200 on origin/main as well, so
# the pin was unsatisfiable on BOTH branches and the merge was never going
# to fix it. The English catalog already agrees with the sources exactly --
# live and reviewed sets are identical, 0 either way -- so the catalogs were
# regenerated correctly and only the ratchet was left behind, which is the
# right way round for this file's "catalogs first, ratchet second" rule.
#
# MOVED 2026-09-12, and this one is 386's Embeddings module arriving rather
# than anything drifting. UI 2,841 -> 2,856 is +18/-3, and MODULE_SUMMARIES
# 67 -> 68 is the `embeddings` module summary itself -- the same shape the
# OPS module made in 6aa5a73aa, where a new module moves the summary count
# by exactly one. The other three tables do NOT move: 388's bystander pair
# and 364's audit were already banked in b5f367f47, and `canonical_sources`
# still returns 1,073/1,068/200 for them.
#
# Measured the way the note above demands, by recomputing the canonical
# identities at 76921939b -- the commit that last moved UI -- and at the
# merge, then diffing the sets. All 21 members, named:
#
#   THE EMBEDDINGS SCREEN, fifteen records added by 067a0a5a7 and 7f0f74ac7
#   (the tile that was in the registry and in none of the tables drawing
#   it). Its controls: "Embed", "Encode every object", "Backbone:",
#   "Batch:", "Channels:", "Per channel (one pass per stain)", "Project to
#   three (one pass)", "Load crops first", "Load crops first." and "no
#   crops loaded". Its prose: the four tooltips that explain the backbone
#   choice, the batch size, what per-channel encoding does to the column
#   names, and what the module is for, plus the warning that a single
#   dimension is not a phenotype.
#
#   The tooltip and the status line really are two records, "Load crops
#   first" and "Load crops first.", differing only in the full stop. That
#   is a duplicate the source should collapse, not something this file can
#   fix: the nine catalogs already carry both, and dropping one here would
#   leave a translated row with no source behind it.
#
#   REWORDED, so one removal and one addition each and no net change:
#     The two normalisation percentile tooltips, which now say what the
#     sixth decimal buys -- 0.0001 clips only the darkest few pixels of a
#     megapixel field, 99.9999 clips a handful of hot pixels where 99.99
#     clips four hundred (7f0f74ac7).
#     The Path-mode tooltip in Preferences, which gained a third paragraph
#     for 327's tour mode now that the twenty coordinates are actually
#     wired to a camera (ac7efdbd4). Same control, new English source,
#     therefore a new identity.
#
# Catalogs first, ratchet second, as ever: all nine locales carry all 18 new
# UI rows and the `embeddings` summary as of 0f3c6dace, and "Embeddings" has
# its nine `_ROWS` translations. Verified before these numbers were touched.
# 2026-09-13. Moved after a RECORD-BY-RECORD REVIEW against 8e9bd0b84, the
# commit each of these counts was last set at -- and that mattered: the counts
# had been edited piecemeal, so `SETTING_LABELS` was last moved at b5f367f47
# while `UI` was last moved at 8e9bd0b84, and differencing against the wrong
# one manufactured a 15-row discrepancy that does not exist. Every count below
# reproduces exactly at 8e9bd0b84.
#
#   SETTINGS      1073 -> 1055      35 removed, 17 added
#   UI            2856 -> 2988       0 removed, 132 added
#   CATEGORY_HELP, MODULE_SUMMARIES  unchanged
#
# ALL 35 REMOVALS ARE `WITHDRAWN_SETTING_SUFFIXES`, none unexplained: the
# intensity band retired across seven roles -- `area_multiplier`,
# `intensity_percentile`, `intensity_threshold_method`,
# `min_intensity_percentile`, `max_intensity_percentile`. `object_roles.py`
# carries the reason: the band "dropped its share of objects however bright
# the field, which is a quota rather than a filter".
#
# THE 17 ADDITIONS ARE ITS REPLACEMENTS AND TWO FEATURES. Seven
# `<role>_intensity_threshold` -- one absolute threshold where the withdrawn
# pair chose between a mean and a percentile; four `organelle*_background` and
# four `organelle*_signal_to_noise` from the organelle channel expansion;
# `remove_background_organelle`; and `barcode_set` for Map Barcodes.
#
# ALL 132 UI ADDITIONS RESOLVE TO A SOURCE FILE IN `spacr/qt`, none orphaned.
# 24 are Map Barcodes. The other 108 are spread over twenty files that each
# define a `_set_status` wrapper, and they are new to this table only because
# the extractor could not see through that wrapper until today -- the strings
# themselves have been on screen all along. The heaviest are convert (11),
# batch (10), foreign (10), model_zoo (9), hyperparam (9), db_browser (7).
#
# Nothing in this move is a caption whose origin is unknown, which is the
# question this ratchet exists to force.
#
# MOVED 2026-09-14 for 364's `grna` retirement, and the interesting number is
# the one that DID NOT move. Measured by set difference against the counts
# above, not by accepting a new total:
#
#   SETTING_LABELS    1055 -> 1054   -1 / +0   the `grna` key
#   SETTING_TOOLTIPS  1050 -> 1049   -1 / +0   the same key's tooltip
#   UI                3291 -> 3291   -1 / +1   AND THIS IS NOT "NO CHANGE"
#   CATEGORY_HELP, MODULE_SUMMARIES  unchanged
#
# THE UI ROW SWAPPED IDENTITY UNDER A CONSTANT TOTAL, which is the exact shape
# this file exists to refuse to read off a total. Two things happened at once:
#
#   OUT: 'Choose the gRNA CSV', the `PATH_LIST_TITLES` file-chooser caption for
#   `grna`. It went with the setting -- a dialog title for a control that is no
#   longer drawn is a caption nine locales still carry and nobody can reach.
#
#   IN: 'gRNA'. This one arrives BECAUSE of the removal, not despite it.
#   'gRNA' is in `build_i18n_catalogs._IDENTITY_TEXT`, and line 3867 does
#   `ui_sources.update(_IDENTITY_TEXT - already_materialized)`. While `grna`
#   was a setting whose English LABEL was 'gRNA', the term was materialised by
#   `set(labels.values())` and therefore excluded from the UI sources. Retiring
#   the setting un-materialises it, so the identity falls through to UI and
#   every catalog needs a UI row for it. Without that row
#   `test_runtime_catalogs_resolve_all_reviewed_false_friend_variants` raises
#   KeyError: 'gRNA' and the standalone-identity test fails with it.
#
# So a NET ZERO here is a removal and an arrival, and the digest below moves
# even though no count does. Catalogs first, ratchet second: all ten carry the
# new 'gRNA' identity row and have lost the four `grna` rows and the chooser
# caption before these numbers were touched.
# MOVED 2026-09-15 for the 08:15 integration batch (wip/integ-0815), measured
# by set difference against 8f4171fd9 -- the commit where every number in this
# dict and the digest below still reproduce byte for byte -- not by accepting a
# new total:
#
#   SETTING_LABELS    1054 -> 1055   +1 / -0   `segmentation_backend` (404/405)
#   SETTING_TOOLTIPS  1049 -> 1050   +1 / -0   the same key's tooltip
#   UI                3291 -> 3445   +154 / -0
#   CATEGORY_HELP, MODULE_SUMMARIES  unchanged
#
# ALL 154 UI ADDITIONS ARE 394 (3d8a269f8), and none is a new string. 394 gave
# the extractor keyed rules for captions passed through local helpers, so
# these were on screen all along and read English in all nine locales. Every
# one is already present at a83a0cfea, the merge of wip/63-394-source before
# any other change of this batch, and neither setting row is. By the file
# under spacr/qt that draws each one:
#
#   volcano_explorer 50, annotate 11, methods_export 9, fast_plots 9,
#   prerun 8, preferences 6, annotation_strategy_panel 6,
#   regression_results 6, hit_list 5, run_history 5, data_manager 4,
#   map_barcodes 4, pipeline_graph 4, settings_search 4, gene_panel 4,
#   percentile_pair 3, save_figure_dialog 3, app_screen 2, run_compare 2,
#   formula_editor 2, hyperparam 1, make_masks 1, annotation_umap_tab 1,
#   measurement_scan_panel 1, object_grid_binding 1, refit_dialog 1,
#   sweep_runs 1
#
# Nothing left the table. The reverse check: removing these 156 identities
# from the current set gives f576cb08... back exactly.
#
# Feature 418, measured against 0f21cbfb3's English catalog on 2026-09-15:
# SETTING_LABELS 1003 -> 982 and SETTING_TOOLTIPS 998 -> 977, each +14/-35.
# The exact role set is cell, nucleus, pathogen, organelle, organelleb,
# organellec, organelled. Each loses minimum_area_to_split,
# min_watershed_distance, intensity_threshold, intensity_merge, intensity_split;
# each gains min_intensity and max_intensity. Numbered slots beyond these
# seven catalogued roles still use the existing runtime registry expansion.
#
# CATEGORY_HELP 194 -> 193, +1/-2: INTENSITY HANDLING's explanation leaves,
# and ADVANCED SETTINGS changes from intensity-driven splitting/merging to
# area, own-channel mean intensity, border filtering and perimeter merging.
# Each explanation is keyed by its full source, so the rewritten umbrella
# contributes one arrival and one removal in both CATEGORY_HELP and UI.
#
# UI 3556 -> 3547, +3/-12: the same two old explanations leave and the new
# umbrella arrives, alongside the new "Min intensity" and "Max intensity"
# captions. The ten other removals are exactly "Intensity Handling (all objects)",
# "Min object area", "Min distance", "Area multiplier", "Min intensity pct",
# "Max intensity pct", "Intensity percentile", "Intensity threshold",
# "Intensity merge", and "Intensity split". These dead helper captions
# were frozen in the builder; COMPARTMENT_FIELDS now supplies current rows.
# MODULE_SUMMARIES remains 68. Total identity delta: +32/-84.
#
# Fourteen existing tooltip identities also have new source text:
# cell_max_area, pathogen_max_area, and each of the four catalogued organelle
# slots' perimeter_fraction, remove_border and remove_border_objects. They
# retain their keys and therefore do not change this identity fingerprint.
# This is a measured source inventory, not verification of translations.
# Catalog equality and locale-quality checks remain pending the catalog pass.
EXTERNAL_SOURCE_COUNTS = {
    # 2026-09-15, the old OPS engine deleted (372): -116 / +0 by SET
    # DIFFERENCE of the identities against the tree before the deletion,
    # and nothing else moved. 52 SETTING_LABELS and 52 SETTING_TOOLTIPS for
    # the settings only the old engine read, and the six OPS category
    # explanations it alone used, which count once under CATEGORY_HELP and
    # once under UI. Nightly 17172faa8's identities minus those 116 digest
    # to the fingerprint below.
    # `recursive` keeps its row: its English now comes from
    # spacr.external_masks, which reads it, so its identity is unchanged.
    "SETTING_LABELS": 982,
    "SETTING_TOOLTIPS": 977,
    # 192 -> 201 on 2026-09-08, +9/-0: the nine OPS section headings that
    # fold onto Align & Stitch. Each needed a curated CATEGORY_TOOLTIPS
    # entry or its panel drew the generic fallback -- a heading whose
    # tooltip says nothing about the settings under it, which costs the
    # reader the hover and tells them nothing. 201 -> 200 on 2026-09-11
    # with `save_to_db`, whose help text was one of them.
    "CATEGORY_HELP": 193,
    # 2,988 -> 3,291 on 2026-09-14, and reviewed record by record against
    # 49c1189f7, where every count in this dict still reproduces exactly.
    # +304 / -1, NOT a flat +303: the four other tables did not move at all,
    # so the whole delta is UI and the single removal is the part a net figure
    # would have hidden.
    #
    #   THE ONE REMOVAL is 'Plate queue' -- sentence case -- which became
    #   'Plate Queue' when the nine _HELP_MODULES display names were
    #   normalised to title case. The row did not leave the interface; it
    #   changed spelling, and it appears among the 304 additions under its new
    #   one. A net +303 reads as 303 arrivals and says nothing about a caption
    #   silently losing its translation to a re-cased key.
    #
    #   THE 304 ADDITIONS are captions that were always on screen and never in
    #   a catalog: 283 reached the extractor through fourteen runtime
    #   registries an AST walk could only see as a variable, and 21 are the
    #   regression-model menu, which `settings_model` DECLARED for exactly
    #   this purpose and which nothing consumed -- so all 21 read English in
    #   all nine locales.
    #
    # 3,291 -> 3,308 on 2026-09-15, +17/-0, and only UI moved: the live
    # magnifier in Make Masks (407) -- its tool-row button, card title, the
    # Classical/Cellpose and Clip/Replace choices ('Skip' is a compact row),
    # 'Updating…', four status lines and six tooltips. Every one reaches the
    # catalog through setText/setToolTip/addItem literals in make_masks.py,
    # and each has a reviewed record in all nine languages except the name
    # 'Cellpose'. Five status TEMPLATES with values filled in are not
    # extracted at all; they are item 65's helper problem, not new rows.
    #
    # 3,308 -> 3,462 on 2026-09-15, +154/-0, when wip/integ-0815 was rebased
    # onto nightly 2d530a813: the 154 rows 394's keyed extractor rules found
    # (enumerated in the note over this dict) join the 17 above. The two sets
    # are DISJOINT, so the count is the plain sum -- and it was MEASURED, not
    # added: canonical_sources() on the rebased tree returns 3,462, so 394's
    # rules found nothing more in the magnifier's new code.
    #
    # 3,462 -> 3,472 on 2026-09-15, +10/-0, only UI, for item 286: the five
    # performance-level tooltips and the five hardware notes. The extractor
    # iterated PERFORMANCE_NOTES and HARDWARE_NOTES as dicts, so it collected
    # their keys ("laptop", ...) and never the prose; it now collects the
    # values too (the keys stay, so no existing row leaves). Each of the ten
    # has a reviewed record in all nine languages, and a plain rebuild changed
    # no other row in any catalog.
    # 3,472 -> 3,466 when the old OPS engine was deleted (372): its six category
    # explanations, which also count under CATEGORY_HELP.
    #
    # 3,466 -> 3,484 on 2026-09-15, +21/-3, and only UI moved: the magnifier's
    # border option and whole-image mode (407), rebased onto nightly
    # df1216b3f. ARRIVING: the Exclude-objects-touching-the-box-border
    # checkbox and its tooltip, the Segment row label with its choices 'Region
    # under the mouse' and 'Whole image' and their tooltip, the new card
    # subtitle, the reworded Size and Magnifier-button tooltips, and eleven
    # status lines -- four of them TEMPLATES ({n}, {error}, {label}) that
    # reach the extractor because they are written as tr("...", n=...) rather
    # than as f-strings. LEAVING: the previous wordings of that subtitle and
    # those two tooltips, which the whole-image mode made untrue. 'Cancel' is
    # a compact row and is not counted here. MEASURED, not added:
    # canonical_sources() on the rebased tree returns 3,484 = 3,466 + 21 - 3.
    #
    # 3,484 -> 3,505 on 2026-09-15, +21/-0, and only UI moved: 387's
    # Dose-Response screen, rebased onto nightly 160380b8c. Eight of the grid's
    # headers and status words (Group, Doses, CI low, CI high, Lack-of-fit p,
    # fitted, unbounded, refused; Status was already a row) -- registered through
    # _DOSE_RESPONSE_UI_SOURCES because they reach their widgets through a
    # tuple and a dict), "all rows", "Fit curve", the Host response and
    # Second compound pickers with their tooltips, "Bliss independence" and
    # "Loewe additivity", and five status lines, four of them TEMPLATES
    # ({name}, {reason}, {rows}, {columns}). Six carry reviewed records in all
    # nine languages (2026-09-15-dose-response-host-readout.json and
    # -combination.json); the other fifteen and the catalogs themselves wait
    # for the pre-release catalog pass, so the English-catalog equality below
    # stays red until it runs. MEASURED, not added: canonical_sources() on the
    # rebased tree returns 3,505, and 3,484 on nightly 160380b8c.
    #
    # 3,505 -> 3,506 on 2026-09-15, +4/-3, for runtime pass A on nightly
    # 5a9c4563a, which is also the catalog pass the note above waits for: the
    # catalogs are regenerated with it, so the English-catalog equality below
    # is green again. ARRIVING: OPS_TOGGLE_TOOLTIP, Mask Generation's OPS
    # switch help, in no catalog until now because AppScreen passes it to
    # AiToggleLabel by a name imported from `mask` (it joins the two toggles
    # built the same way in _indirect_runtime_ui_sources), and the rewritten
    # OPS INPUT / ALIGNMENT / PERFORMANCE explanations. LEAVING: those three
    # explanations' mosaic-era wording, which described the deleted engine.
    # The same three swap under CATEGORY_HELP, so its count stays 194; the
    # other seven OPS strings rewritten with them are keyed by setting or
    # module name and move no identity. MEASURED: canonical_sources() returns
    # 3,506 = 3,505 + 4 - 3.
    #
    # 3,506 -> 3,527 on 2026-09-15, +21/-0, for runtime pass B on nightly
    # 08a2c1719 (wip/api-pass-412-416-413): Make Masks' "Load test data…"
    # tooltip and its five status strings (412) and the in-app update's
    # fifteen dialog strings and removal reasons (416), each with a reviewed
    # record in all nine locales. Nothing leaves. MEASURED:
    # canonical_sources() returns 3,527 = 3,506 + 21.
    # 3527 -> 3541 on 2026-09-15, +14/-0 by SET DIFFERENCE: the
    # Dose-Response sources the work session added in 978092685 -- the
    # two Z' plate verdicts (usable, refused), the pooled-EC50,
    # selectivity-index and synergy-excess lines with their four
    # refusal captions, the plates-disagree line, the axis words
    # "concentration" and "response", and the whole-table note.
    # 417: 3,541 -> 3,556, +20/-5 against a2ecc32b8's English manifest.
    # Eighteen authored strings (12 settings/model rows and 6 drag rows)
    # plus the product names DINOCell/SAMCell arrive; five old tooltips leave.
    # Every new prose row has a reviewed record in each of the nine locales.
    # The runtime pass preserved every pre-existing translated value.
    "UI": 3547,
    "MODULE_SUMMARIES": 68,
}
# Moved with the counts above. The identity that changed is one UI row: the
# invented-negatives notice replaced "{n} outstanding suggestions thrown away
# before fitting.", which is no longer anywhere in the source.
#
# Moved again on 2026-09-08 with no count change: the shortcuts help row is
# keyed by its English source, and that source now says Ctrl+P where it said
# Ctrl+comma. Same row, new identity.
#
# Moved again on 2026-09-08 with the CATEGORY_HELP count above: nine new
# section headings are nine new record identities.
#
# Moved again on 2026-09-11 with the counts above: 26 record identities
# change, which is more than the net -4 suggests because seven of the
# removals are renames and arrive back under a new key. Enumerated in the
# note over EXTERNAL_SOURCE_COUNTS.
# Moved again on 2026-09-12 with the counts above: 21 record identities
# change, 18 arriving and 3 leaving, enumerated in the note over
# EXTERNAL_SOURCE_COUNTS.
# Moved again on 2026-09-13 with the counts above: 184 record identities
# change, 149 arriving and 35 leaving, every one classified in the note over
# EXTERNAL_SOURCE_COUNTS.
# Moved again on 2026-09-14 for `grna`, and THIS TIME THE DIGEST IS THE ONLY
# WITNESS TO PART OF THE CHANGE. Four identities move: ('SETTING_LABELS',
# 'grna') and ('SETTING_TOOLTIPS', 'grna') leave, ('UI', 'Choose the gRNA CSV')
# leaves and ('UI', 'gRNA') arrives. The UI pair cancels in the count and does
# not cancel here -- which is the point of pinning identities and not just
# totals.
# Moved again on 2026-09-15 with the UI count above, for the live magnifier
# (407): 17 record identities change, 17 arriving and 0 leaving, all UI:
#
#   ('UI', 'Cellpose')        ('UI', 'Classical')      ('UI', 'Clip')
#   ('UI', 'Replace')         ('UI', 'Magnifier')      ('UI', 'Live magnifier')
#   ('UI', 'Updating…')
#   ('UI', 'Magnifier off. The objects it added stay in the mask.')
#   ('UI', 'Magnifier on: a click adds the objects outlined in the box; ...')
#   ('UI', 'Magnifier: nothing to add — the box outlines no object, ...')
#   ('UI', 'Segments the region under the mouse. A click adds the ...')
#   ('UI', 'How many times larger than the canvas the box draws ...')
#   ('UI', 'How readily an object is accepted. Raise it to take in ...')
#   ('UI', 'Show a box under the mouse with the region around it ...')
#   ('UI', 'Side of the square region the model segments, in image ...')
#   ('UI', 'What a new object does where the mask already has an ...')
#   ('UI', 'Which model segments the region in the box. Classical ...')
#
# PROVED, NOT ACCEPTED: the digest recomputed with this test's own formula
# over today's identities MINUS those seventeen is
# f576cb08f184154f97bb47c8e8fb2af1013f17efdb419a70ad825093710f0dcf, the
# previous pin byte for byte, and so is the digest over the identities of the
# committed en.py before the rebuild. Nothing else arrived and nothing left;
# the fourteen label corrections that rode with these records changed
# translations of existing rows, which are values, not identities.
# Moved again on 2026-09-15 with the counts above: 156 identities arrive and
# none leave -- 154 UI rows from 394 and ('SETTING_LABELS',
# 'segmentation_backend') / ('SETTING_TOOLTIPS', 'segmentation_backend').
# Moved again on 2026-09-15 by the rebase of wip/integ-0815 onto nightly
# 2d530a813, which joins the two moves above: 173 identities arrive (the 17
# magnifier rows and the 156 of this batch) and none leave. PROVED BY
# SUBTRACTION with this test's own formula over the rebased tree's identities:
#
#   all identities                     ba2a0af05393208f...  (the pin below)
#   minus the 17 magnifier rows        dba7b70df0f039c3...  this batch's pin
#   minus the 156 of this batch        dacb857799ad804e...  nightly's pin
#   minus both                         f576cb08f184154f...  the 8f4171fd9 pin
# Moved again on 2026-09-15 for item 286, with the UI count above: 10
# identities arrive and none leave, all UI -- the five performance-level
# tooltips (PERFORMANCE_NOTES) and the five hardware notes (HARDWARE_NOTES).
# PROVED BY SUBTRACTION with this test's own formula: today's identities
# minus those ten give ba2a0af05393208f..., the previous pin byte for byte.
# Moved again on 2026-09-15 when the old OPS engine was deleted (372): 116
# identities leave and none arrive, the same set the counts above name.
# PROVED BY SUBTRACTION the same way: nightly's identities (119746de...)
# minus those 116 give this pin byte for byte.
# Moved again on 2026-09-15 with the UI count above, for the magnifier's
# border option and whole-image mode (407), rebased onto nightly df1216b3f:
# 24 identities change, 21 arriving and 3 leaving, all UI, enumerated in the
# note over EXTERNAL_SOURCE_COUNTS. PROVED BY SUBTRACTION with this test's own
# formula over the rebased tree: today's identities (520f3df5...) minus the
# 21 arrivals plus the 3 retired wordings give
# 3bc7769a7d287175045d24611bbd962d16623d68f52ae9101aa59602d57669e3, the base's
# pin byte for byte.
# Moved again on 2026-09-15 with the UI count above, for 387's Dose-Response
# screen, rebased onto nightly 160380b8c: 21 identities arrive and none leave,
# all UI, named in the note over EXTERNAL_SOURCE_COUNTS. PROVED BY
# SUBTRACTION with this test's own formula: today's identities minus those
# 21 give 520f3df5..., the previous pin byte for byte.
# Moved again on 2026-09-15 for runtime pass A on nightly 5a9c4563a, with the
# UI count above: 13 identities change, 7 arriving (UI: the OPS toggle tooltip
# and the three rewritten OPS category explanations; CATEGORY_HELP: the same
# three) and 6 leaving (their mosaic-era wordings, under both tables). PROVED
# BY SUBTRACTION with this test's own formula: today's identities
# (6408b9e4...) minus the 7 arrivals plus the 6 leavers give
# b334316a64fc5ca9f34d6f5b73836b704834d19f88734a962e2ede181d32bd54, the
# previous pin byte for byte.
# Moved again on 2026-09-15 for runtime pass B on nightly 08a2c1719, with the
# UI count above: 21 identities arrive, all UI (412's six and 416's fifteen
# captions), and none leave. PROVED BY SUBTRACTION with this test's own
# formula: today's identities (01ae52fc...) minus those 21 give
# 6408b9e46d4b7478430257a6f632bbb12ec43a279df807213c4433b8e37d6a72, the
# previous pin byte for byte.
EXTERNAL_SOURCE_KEY_SHA256 = (
    # 418: 0f21cbfb3's English identities reproduce the previous pin exactly:
    # b4d1896bbc1135f9f4098b3473ac4163d9cf36bf3d58c7447daeb652dacf725f.
    # The +32/-84 identities named above give this current source digest.
    "476736243fbfe6359464c3ce219efe4a66f17dbec4f596710050a29595b9a5ac"
)

# Calls whose literal argument is chrome owned by the compact catalog on the
# onboarding surfaces.  Their registry/data-driven captions are added by
# ``compact_user_facing_captions`` below as well.
_ONBOARDING_LITERAL_CALLS = {
    "QCheckBox",
    "QLabel",
    "QPushButton",
    "Toggle",
    "_say",
    "addButton",
    "setText",
    "setWindowTitle",
    "tr",
}
_ONBOARDING_PATHS = (
    ROOT / "spacr" / "qt" / "widgets" / "setup_slides.py",
    ROOT / "spacr" / "qt" / "first_run.py",
    ROOT / "spacr" / "qt" / "install_consent.py",
)


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _onboarding_literal_captions() -> set[str]:
    """Find independently authored literal chrome on first-run surfaces."""
    found: set[str] = set()
    for path in _ONBOARDING_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name not in _ONBOARDING_LITERAL_CALLS:
                continue
            # ``addButton(caption, role)`` has one caption.  All other calls
            # in this focused set likewise expose their caption first.
            if not node.args:
                continue
            try:
                value = ast.literal_eval(node.args[0])
            except (TypeError, ValueError):
                continue
            if isinstance(value, str) and value.strip():
                found.add(value.strip())
    return found


_CUSTOM_WIDGET_ARGUMENTS = {
    # positional indices, keyword names
    "AiToggleLabel": ((1, 2), {"text", "tooltip"}),
    "Card": ((0, 1), {"title", "subtitle"}),
    "FlatButton": ((0, 2), {"text", "tooltip"}),
    "FlatComboBox": ((1,), {"tooltip"}),
    "FlatSpinBox": ((1,), {"tooltip"}),
    "Toggle": ((0,), {"text"}),
}

# Visible technical identities and formatting shells deliberately excluded
# from both translation layers.  Their exact spelling carries a dimensional,
# backend, field-name or interpolation contract; the separate assertion below
# prevents a term fallback from rewriting them.
_RUNTIME_IDENTITY_CAPTIONS = {
    "%d px",
    "3D",
    '<a href="api">API</a>',
    "CPU",
    "Cellpose-SAM",
    "GPU",
    "MIP",
    "RdBu_r",
    "image_path",
    "metadata_column_map.json",
    "png_list",
    "png_path",
    "spaCR",
    "{report}",
    "■ {note}",
}


def _static_string(node: ast.AST, constants: dict[str, ast.AST]) -> str | None:
    """Resolve one literal custom-widget argument without using the builder."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id in constants:
        return _static_string(constants[node.id], constants)
    if (
        isinstance(node, ast.Call)
        and _call_name(node) == "tr"
        and node.args
    ):
        return _static_string(node.args[0], constants)
    return None


def _custom_widget_literal_captions() -> set[str]:
    """Independently find literal text carried by spaCR widget constructors."""
    found: set[str] = set()
    for path in sorted((ROOT / "spacr" / "qt").rglob("*.py")):
        if "i18n_catalogs" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        constants: dict[str, ast.AST] = {}
        for statement in tree.body:
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                constants[statement.targets[0].id] = statement.value
            elif (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.value is not None
            ):
                constants[statement.target.id] = statement.value
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            contract = _CUSTOM_WIDGET_ARGUMENTS.get(_call_name(node))
            if contract is None:
                continue
            positions, keywords = contract
            candidates = [
                node.args[index] for index in positions if index < len(node.args)
            ]
            candidates.extend(
                keyword.value
                for keyword in node.keywords
                if keyword.arg in keywords
            )
            for candidate in candidates:
                value = _static_string(candidate, constants)
                if value is not None and re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]{2,}", value):
                    found.add(value.strip())
    return found


def _indirect_registry_captions() -> set[str]:
    """Independently enumerate captions passed through data registries."""
    from spacr.qt.preferences import (
        MODE_LABELS,
        MODE_NOTES,
        MODE_WARNINGS,
        PREFERENCE_TIPS,
    )
    from spacr.qt.preview_registry import PREVIEWS
    from spacr.qt.screens.app_screen import DIMENSION_TOGGLES
    from spacr.qt.screens.batch import ON_ERROR_LABELS
    from spacr.qt.screens.hyperparam import TOGGLE_TEXT, TOGGLE_TOOLTIP
    from spacr.qt.screens.parameter_sweep import (
        SWEEP_TOGGLE_TEXT,
        SWEEP_TOGGLE_TOOLTIP,
    )
    from spacr.qt.widgets.ambient import (
        ANIMATION_CHOICES,
        DRIFT_DIRECTIONS,
        PALETTE_SETS,
        animation_label,
        animation_note,
        drift_direction_label,
        drift_direction_note,
    )
    from spacr.qt.widgets.preview_contract import (
        PREVIEW_BUSY_MESSAGE,
        PREVIEW_CANCEL_TEXT,
        PREVIEW_CANCELLED_MESSAGE,
        PREVIEW_RUN_TEXT,
        PREVIEW_RUNNING_MESSAGE,
        PRIMARY_NOTES,
    )
    from spacr.qt.widgets.preview_controls import (
        ALL_CHANNELS,
        MAX_SETS_TOOLTIP,
    )

    found = set(PREFERENCE_TIPS) | set(PREFERENCE_TIPS.values())
    found.update(MODE_LABELS.values())
    found.update(MODE_NOTES.values())
    found.update(value for value in MODE_WARNINGS.values() if value)
    found.update(label for label, _value in ON_ERROR_LABELS)
    for _dimension, label, tooltip in DIMENSION_TOGGLES:
        found.update((label, tooltip))
    found.update((
        TOGGLE_TEXT,
        TOGGLE_TOOLTIP,
        SWEEP_TOGGLE_TEXT,
        SWEEP_TOGGLE_TOOLTIP,
        ALL_CHANNELS,
        MAX_SETS_TOOLTIP,
        PREVIEW_RUN_TEXT,
        PREVIEW_CANCEL_TEXT,
        PREVIEW_BUSY_MESSAGE,
        PREVIEW_CANCELLED_MESSAGE,
        PREVIEW_RUNNING_MESSAGE,
        "Preview failed: {error}",
        "Channels drawn in {mode} primaries.",
    ))
    found.update(PRIMARY_NOTES.values())
    for spec in PREVIEWS.values():
        found.add(spec.title)
        found.add(
            spec.tooltip
            or "Show a preview of what these settings produce."
        )
    for name in ANIMATION_CHOICES:
        found.update((animation_label(name), animation_note(name)))
    for spec in PALETTE_SETS.values():
        found.update((spec.label, spec.note))
    for name in DRIFT_DIRECTIONS:
        found.update((
            drift_direction_label(name),
            drift_direction_note(name),
        ))
    return {str(value).strip() for value in found if str(value).strip()}


def _shortcut_caption_fields() -> set[str]:
    """Return shortcut copy while deliberately excluding key identifiers."""
    from spacr.qt.shortcuts import SCREEN_SHORTCUTS, SHORTCUTS

    return {
        value.strip()
        for spec in (*SHORTCUTS, *SCREEN_SHORTCUTS)
        for value in (spec.label, spec.category, spec.scope)
        if value.strip()
    }


def compact_user_facing_captions() -> frozenset[str]:
    """Discover the complete caption surface that exact ``_ROWS`` owns.

    This predicate intentionally follows semantic UI registries rather than
    reading the catalog it audits: Home app/section names, fold buttons,
    first-run slides/questions/choices, tour text, terms chrome and literal
    onboarding dialog/button captions and shortcut labels, categories and
    scopes.  A caption added to any of those sources therefore enters this
    set before it has a translation row.  Shortcut key identifiers are not
    copy and remain in their platform-native spelling.

    Language choices remain in their own native scripts and installed AI
    provider names remain product identities.  The generic provider fallback
    is prose, so it is included.  Setting help, category help, module
    summaries and other static Qt prose are generated-catalog candidates and
    are covered by :func:`test_every_generated_catalog_candidate_has_a_source_hash`.
    """
    import spacr.qt
    from spacr.qt.app import APPS
    from spacr.qt.first_run import DEFAULT_TOUR
    from spacr.qt.setup_screen import questions
    from spacr.qt.terms import TRANSLATIONS, register_translations
    from spacr.qt.widgets.ambient import (
        ANIMATION_CHOICES,
        animation_label,
    )
    from spacr.qt.widgets.fold_strip import folded_modules
    from spacr.qt.widgets.setup_slides import ANIMATION_LABEL, SLIDES

    # Terms and late app registrations use the same supported registration
    # seam as plugins.  Registering is idempotent and makes the relationship
    # with the runtime ``_ROWS`` object deterministic in any import order.
    # Calling the complete self-registration seam here is also an import-order
    # regression test: a compact fold row that disagrees with its host app's
    # registered metadata raises instead of being hidden in the startup log.
    spacr.qt.register_self_registering_modules()
    register_translations()

    found = _onboarding_literal_captions()
    found.update(name for _key, name, _desc, _section in APPS)
    found.update(section for _key, _name, _desc, section in APPS)
    found.update(entry[0] for entry in folded_modules().values())
    found.update(
        text
        for title, blurb, _keys in SLIDES
        for text in (title, blurb)
    )
    found.add(ANIMATION_LABEL)
    found.update(question[1] for question in questions())
    for key, _label, _getter, _setter, choices in questions():
        if key == "language":
            # Native names are identity labels for readers who cannot yet
            # read the selected UI language.
            continue
        if key == "ai_provider":
            # Product names/logos stay exact.  The empty-value fallback is a
            # sentence fragment and is therefore translated.
            found.update(
                label for value, label in (choices or ()) if not value
            )
            continue
        found.update(label for _value, label in (choices or ()))
    found.update(animation_label(name) for name in ANIMATION_CHOICES)
    found.update(
        text
        for step in DEFAULT_TOUR
        for text in (step.title, step.body)
    )
    found.update(_shortcut_caption_fields())
    found.update(source for source, _values in TRANSLATIONS)
    # DELIBERATE TECHNICAL IDENTITIES ARE NOT PART OF THIS SURFACE.
    #
    # `_RUNTIME_IDENTITY_CAPTIONS` already held GPU and is asserted separately
    # by `test_runtime_identity_captions_remain_exact_in_every_language`, but
    # this function did not subtract it -- so GPU was simultaneously required
    # to stay exact in every language AND required to have a translated
    # `_ROWS` row. CPU joined it on 2026-09-02 when the maintainer answered
    # 316-A: "Keep exact", the same reasoning instruction 318 used to hold
    # 'QC' exact in every locale because QC is an identifier.
    return frozenset(found) - _RUNTIME_IDENTITY_CAPTIONS


def test_compact_user_facing_caption_surface_has_exact_rows_and_is_pinned():
    """Every compact caption has one exact ``_ROWS`` row, never a fallback.

    The independent semantic discovery runs before either catalog is read.
    Consequently, adding one of these captions to a generated registry cannot
    make it escape the literal-row requirement: it remains in ``discovered``
    and fails here until an exact row is reviewed.
    """
    from spacr.qt.i18n import _ROWS

    discovered = compact_user_facing_captions()
    missing = sorted(discovered - set(_ROWS))
    assert not missing, (
        "new compact user-facing captions need exact i18n._ROWS entries:\n  "
        + "\n  ".join(repr(value) for value in missing)
    )
    assert len(discovered) == COMPACT_CAPTION_COUNT, (
        f"compact caption surface changed from {COMPACT_CAPTION_COUNT} to "
        f"{len(discovered)}; add/review exact rows, then move the ratchet"
    )
    digest = hashlib.sha256(
        "\0".join(sorted(discovered)).encode("utf-8")
    ).hexdigest()
    assert digest == COMPACT_CAPTION_SHA256, (
        "compact caption set changed without moving its reviewed fingerprint"
    )


def test_compact_and_generated_caption_owners_are_disjoint():
    """A caption belongs to the reviewed compact or generated layer, not both."""
    from spacr.qt.i18n import _ROWS
    from spacr.qt.i18n_catalogs import en

    duplicated = sorted(set(_ROWS) & set(en.UI_SOURCES))
    assert not duplicated, (
        "compact captions must not acquire a second generated owner:\n  "
        + "\n  ".join(repr(value) for value in duplicated)
    )


def test_shortcut_copy_enters_the_compact_ratchet_but_keys_do_not():
    """Shortcut labels/categories/scopes are copy; bindings are identities."""
    from spacr.qt.i18n import _ROWS, VALID_LANGUAGE_CODES, tr
    from spacr.qt.shortcuts import SCREEN_SHORTCUTS, SHORTCUTS

    discovered = compact_user_facing_captions()
    assert _shortcut_caption_fields() <= discovered

    key_identifiers = {
        spec.keys for spec in (*SHORTCUTS, *SCREEN_SHORTCUTS)
    }
    assert not (key_identifiers & set(_ROWS)), (
        "shortcut bindings must retain QKeySequence/native platform spelling"
    )
    for language in VALID_LANGUAGE_CODES[1:]:
        annotate = tr("Annotate", language)
        make_masks = tr("Make Masks", language)
        joint_scope = tr("the Annotate and Make Masks screens", language)
        assert annotate in joint_scope
        assert make_masks in joint_scope
        assert annotate in tr("the Annotate screen", language)
        assert make_masks in tr("the Make Masks screen", language)


def test_shortcut_overlay_renders_localized_copy_and_native_keys(
    monkeypatch,
    qtbot,
):
    """A newly opened map follows the language without translating bindings."""
    from PySide6.QtWidgets import QLabel, QWidget

    from spacr.qt.shortcuts import ShortcutOverlay, native

    expected = {
        "sv": ("Pensel  —  skärmen Skapa masker", "SKAPA MASKER"),
        "ko": ("브러시  —  마스크 만들기 화면", "마스크 만들기"),
    }
    for language, (brush_text, category_text) in expected.items():
        monkeypatch.setenv("SPACR_LANGUAGE", language)
        window = QWidget()
        window.resize(1400, 900)
        qtbot.addWidget(window)
        overlay = ShortcutOverlay(window)

        labels = overlay.findChildren(QLabel)
        copy = {
            label.text()
            for label in labels
            if label.objectName() == "ShortcutOverlayLabel"
        }
        categories = {
            label.text()
            for label in labels
            if label.objectName() == "ShortcutOverlayCategory"
        }
        keys = {
            label.text()
            for label in labels
            if label.objectName() == "ShortcutOverlayKeys"
        }

        assert brush_text in copy
        assert category_text in categories
        assert native("B") in keys


def test_compact_rows_are_unique_and_dynamic_registries_are_unambiguous():
    """Reject silent dict duplicates and two translations for one caption."""
    path = ROOT / "spacr" / "qt" / "i18n.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keys: list[str] = []
    for statement in tree.body:
        if (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "_ROWS"
            and isinstance(statement.value, ast.Dict)
        ):
            keys = [ast.literal_eval(key) for key in statement.value.keys]
            break
    duplicates = sorted(
        key for key, count in Counter(keys).items() if count > 1
    )
    assert keys, "could not locate the literal _ROWS catalog"
    assert not duplicates, f"duplicate _ROWS keys: {duplicates}"

    from spacr.qt.app import APPS, registered_metadata
    from spacr.qt.i18n import _ROWS, _TERM_ROWS
    from spacr.qt.terms import TRANSLATIONS

    names = {key: name for key, name, _desc, _section in APPS}
    dynamic = [
        (names[key], tuple(values))
        for key, values in registered_metadata("translations").items()
    ]
    dynamic.extend((source, tuple(values)) for source, values in TRANSLATIONS)
    conflicts = [
        source
        for source, values in dynamic
        if source not in _ROWS or _ROWS[source] != values
    ]
    assert not conflicts, f"ambiguous/missing registered rows: {conflicts}"
    overlap_conflicts = sorted(
        source
        for source in set(_ROWS) & set(_TERM_ROWS)
        if _ROWS[source] != _TERM_ROWS[source]
    )
    assert not overlap_conflicts, (
        f"exact and term catalogs disagree for: {overlap_conflicts}"
    )


def test_spanish_compact_rows_use_consistent_formal_register():
    """User instructions must not mix informal Spanish into formal chrome."""
    from spacr.qt.i18n import _ROWS

    spanish = {source: values[2] for source, values in _ROWS.items()}
    informal_pronoun = re.compile(
        r"\b(?:tú|tu|tus|te|ti|contigo|puedes|estés|hayas|veas)\b",
        re.IGNORECASE,
    )
    rejected_phrases = (
        "Abre una incidencia",
        "Activa o desactiva el UMAP",
        "Borra el cuadro",
        "Carga con un clic",
        "Ejecuta spacr-doctor",
        "Genera un conjunto",
        "Haz clic",
        "Pasa el cursor",
        "Pulsa Esc",
        "Suelta una carpeta",
        "También puedes",
        "Usa Demostraciones",
        "cuando pulsas Enviar",
        "ejecútalo en una terminal",
        "elige un conjunto",
        "en tu navegador",
        "introduce {code}",
        "para que veas",
        "si cambias de opinión",
        "selecciona ⓘ",
        "te muestra",
        "Úsalos",
    )
    offenders = {
        source: target
        for source, target in spanish.items()
        if informal_pronoun.search(target)
        or any(phrase.casefold() in target.casefold()
               for phrase in rejected_phrases)
    }
    assert not offenders, f"informal Spanish compact captions: {offenders}"


def test_every_generated_catalog_candidate_has_a_source_hash():
    """The non-compact UI surface is complete in the external registry."""
    tools_dir = str(ROOT / "tools")
    sys.path.insert(0, tools_dir)
    try:
        builder = import_module("build_i18n_catalogs")
    finally:
        sys.path.remove(tools_dir)

    from spacr.qt.i18n_catalogs import en

    candidates = set(builder.extract_static_ui_sources())
    missing_sources = sorted(candidates - set(en.UI_SOURCES))
    assert not missing_sources, (
        "generated UI candidates missing from en.UI_SOURCES:\n  "
        + "\n  ".join(repr(value) for value in missing_sources)
    )
    stale = sorted(
        source
        for source in candidates
        if en.SOURCE_HASHES.get(("UI", source))
        != hashlib.sha256(source.encode("utf-8")).hexdigest()
    )
    assert not stale, (
        "generated UI candidates missing current source hashes:\n  "
        + "\n  ".join(repr(value) for value in stale)
    )


def test_external_caption_layer_is_complete_exclusive_and_pinned():
    """Every non-compact caption has one reviewed source-bound record.

    This is the generated layer's counterpart to the literal ``_ROWS``
    ratchet.  A newly discovered caption cannot pass merely because catalog
    generation notices it: its table/key changes this count or digest and
    requires an explicit review and ratchet update.
    """
    tools_dir = str(ROOT / "tools")
    sys.path.insert(0, tools_dir)
    try:
        builder = import_module("build_i18n_catalogs")
    finally:
        sys.path.remove(tools_dir)

    from spacr.qt.i18n import _ROWS
    from spacr.qt.i18n_catalogs import en

    canonical = builder.canonical_sources()
    external = {
        "SETTING_LABELS": canonical["setting_labels"],
        "SETTING_TOOLTIPS": canonical["setting_tooltips"],
        "CATEGORY_HELP": canonical["categories"],
        "UI": canonical["ui"],
        "MODULE_SUMMARIES": canonical["module_summaries"],
    }
    counts = {table: len(records) for table, records in external.items()}
    assert counts == EXTERNAL_SOURCE_COUNTS, (
        "external caption inventory changed; review every new/removed record "
        f"before moving the ratchet: {counts}"
    )

    identities = sorted(
        (table, str(key))
        for table, records in external.items()
        for key in records
    )
    digest = hashlib.sha256(
        "\0".join(
            f"{table}\0{key}" for table, key in identities
        ).encode("utf-8")
    ).hexdigest()
    assert digest == EXTERNAL_SOURCE_KEY_SHA256, (
        "external caption identities changed without moving their reviewed "
        "fingerprint"
    )

    english = {
        "SETTING_LABELS": en.SETTING_LABELS,
        "SETTING_TOOLTIPS": en.SETTING_TOOLTIPS,
        "CATEGORY_HELP": en.CATEGORY_SOURCES,
        "UI": en.UI_SOURCES,
        "MODULE_SUMMARIES": en.MODULE_SUMMARIES,
    }
    assert {
        table: set(records) for table, records in external.items()
    } == {
        table: set(records) for table, records in english.items()
    }
    assert not (set(_ROWS) & set(external["UI"])), (
        "compact and source-bound caption layers must be disjoint"
    )


def test_custom_widgets_and_indirect_registries_enter_one_i18n_layer():
    """Dynamic/custom captions have exactly one explicit reviewed owner."""
    tools_dir = str(ROOT / "tools")
    sys.path.insert(0, tools_dir)
    try:
        builder = import_module("build_i18n_catalogs")
    finally:
        sys.path.remove(tools_dir)

    from spacr.qt.i18n import _ROWS
    from spacr.qt.i18n_catalogs import CATALOG_LANGUAGES, en

    discovered = set(builder.extract_static_ui_sources())
    independently_expected = (
        _custom_widget_literal_captions() | _indirect_registry_captions()
    )
    missing_ownership = sorted(
        independently_expected
        - _RUNTIME_IDENTITY_CAPTIONS
        - set(_ROWS)
        - discovered
    )
    assert not missing_ownership, (
        "custom/indirect runtime captions bypass both i18n layers:\n  "
        + "\n  ".join(repr(value) for value in missing_ownership)
    )

    ambiguous_ownership = sorted(
        source
        for source in independently_expected - _RUNTIME_IDENTITY_CAPTIONS
        if int(source in _ROWS) + int(source in en.UI_SOURCES) != 1
    )
    assert not ambiguous_ownership, (
        "custom/indirect captions need exactly one compact or generated "
        "owner:\n  "
        + "\n  ".join(repr(value) for value in ambiguous_ownership)
    )

    external = (
        independently_expected - _RUNTIME_IDENTITY_CAPTIONS - set(_ROWS)
    )
    assert external <= set(en.UI_SOURCES)
    for language in CATALOG_LANGUAGES:
        module = import_module(f"spacr.qt.i18n_catalogs.{language}")
        missing = sorted(external - set(module.UI))
        assert not missing, f"{language} lacks indirect UI rows: {missing}"
        stale = sorted(
            source
            for source in external
            if module.SOURCE_HASHES.get(("UI", source))
            != hashlib.sha256(source.encode("utf-8")).hexdigest()
        )
        assert not stale, f"{language} has stale indirect UI hashes: {stale}"
        blank = sorted(source for source in external if not module.UI[source].strip())
        assert not blank, f"{language} has blank indirect UI rows: {blank}"


def test_runtime_identity_captions_remain_exact_in_every_language():
    """Technical names, field names and formatting shells stay exact."""
    from spacr.qt.i18n import VALID_LANGUAGE_CODES, tr

    for language in VALID_LANGUAGE_CODES:
        assert {
            source: tr(source, language)
            for source in _RUNTIME_IDENTITY_CAPTIONS
        } == {source: source for source in _RUNTIME_IDENTITY_CAPTIONS}


def test_dynamic_preview_messages_are_rendered_from_localized_templates(
    monkeypatch,
):
    """Runtime-formatted preview status text must use catalog templates."""
    from spacr.qt import i18n
    from spacr.qt.i18n_catalogs import CATALOG_LANGUAGES
    from spacr.qt.widgets.preview_contract import (
        PREVIEW_BUSY_MESSAGE,
        PRIMARY_NOTES,
        LivePreviewContract,
        preview_failure_message,
    )

    class Label:
        def __init__(self):
            self.value = ""

        def setText(self, value):  # noqa: N802 - minimal Qt label contract
            self.value = str(value)

    class Panel(LivePreviewContract):
        def __init__(self):
            self._status = Label()

        def display_primaries(self):
            return "rgb"

    for language in CATALOG_LANGUAGES:
        monkeypatch.setattr(
            i18n, "current_language", lambda code=language: code
        )
        failure = preview_failure_message("E42")
        expected_failure = i18n.tr(
            "Preview failed: {error}", language, error="E42"
        )
        assert failure == expected_failure
        assert failure != "Preview failed: E42"

        panel = Panel()
        panel.set_preview_status(PREVIEW_BUSY_MESSAGE)
        assert panel._status.value == i18n.tr(
            PREVIEW_BUSY_MESSAGE, language
        )
        assert panel._status.value != PREVIEW_BUSY_MESSAGE

        panel.display_primaries = lambda: "tritanope"
        note = panel.display_primaries_note()
        assert note == i18n.tr(PRIMARY_NOTES["tritanope"], language)
        assert note != PRIMARY_NOTES["tritanope"]
