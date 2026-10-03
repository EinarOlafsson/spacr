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
# 208 -> 243 on 2026-09-25, +39/-4, instruction 316. MEASURED BY SET
# DIFFERENCE against this pin's own commit (89094e3c4), whose tree reproduces
# 4663f4f0... byte for byte. Arrived with rows written here, AI technical
# review (Claude Opus 5.5), no native-speaker signoff: 34 captions -- the
# setup installer and GitHub CLI prompts (13), the ten night-theme names and
# the Resonance backdrop (11), "Test data and walkthroughs" and its body on
# the first-run tour (2), two Help shortcuts, Annotate's "Confirm/Reject the
# suggested label" (item 512), "Install", "Keep installing", "Stop it and
# close" and "Signing in to {label}…". Arrived with their nine rows already
# written by the items that added them (5): "Candida spp.", "Plasmodium
# spp.", "Toxoplasma", "Host–Pathogen Analysis" and the rewritten issue-filing
# privacy note. Retired (4): "Demos menu", its one-click demo blurb,
# "Settings recipes" and the old privacy note. Their `_ROWS` rows stay.
COMPACT_CAPTION_COUNT = 243
COMPACT_CAPTION_SHA256 = (
    "6ad2ef9e9b4495a569d1011d594d21fa93f12c8ff161b4a0776369819db42618"
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
#
# 316, 2026-09-25: the runtime catalog pass. MEASURED by SET DIFFERENCE of
# the identities against c0b2c5227, where the previous counts and digest
# reproduce exactly (476736243f...): +2304/-122 across the five tables,
# SETTING_LABELS 982 -> 1086 (+110/-6), SETTING_TOOLTIPS 977 -> 1108
# (+137/-6), CATEGORY_HELP 193 -> 211 (+21/-3), UI 3547 -> 5472
# (+2032/-107), MODULE_SUMMARIES 68 -> 72 (+4/-0: candida, host_pathogen,
# plasmodium, toxoplasma). These are the features merged between 418 and
# item 530 while no catalog was rebuilt. Every catalog is now regenerated
# without a model from reviewed records, and every identity has a
# source-bound translation in all nine languages except 35 Hindi UI captions
# (first-pass IDs 821-855), whose translator output a safety classifier
# stopped; they show in English and nothing machine-generated replaced them.
# 316, 2026-09-27: all nine regenerated catalogs passed the canonical audit.
# Measured against 6ba9eba0f: +813/-37 identities, including the 11 newly
# extracted captions. Exact diff and review scope are recorded in
# features/data/316_runtime_inventory_delta_2026-09-27.json.
# AI technical review (Codex), no native-speaker signoff.
# 316, 2026-09-28: +19/-0 identities (559: four labels and four tooltips;
# 558: eight UI captions; 556: three SAM2 UI captions). All nine locales
# have source-bound AI technical review, no native-speaker signoff.
# Every existing translation is preserved; exact additions and prior
# fingerprint: features/data/316_runtime_alpha_delta_2026-09-28.json.
# 316, 2026-09-28: +14/-0 identities (564/576: five labels, five tooltips;
# one category help and three UI captions). All nine locales
# have source-bound AI technical review, no native-speaker signoff.
# Every existing translation is preserved; exact additions and prior
# fingerprint: features/data/316_runtime_counterfactual_databases_delta_2026-09-28.json.
# 2026-10-01: measured canonical inventory after the complete nine-locale
# generator audit. 214 identities added and 8 removed relative to the
# preceding committed English catalogue. Translation quality gates unchanged.
# 2026-10-01 N628/N629: measured canonical inventory after the complete nine-locale
# generator audit. 72 identities added and 14 removed relative to the
# preceding committed English catalogue. Translation quality gates unchanged.
# 2026-10-01 F542: measured canonical inventory after the complete nine-locale
# generator audit. One identity added and none removed relative to the
# preceding committed English catalogue. Translation quality gates unchanged.
# 2026-10-02 F583: measured canonical inventory after the complete nine-locale
# generator audit. 13 identities added and 0 removed relative to the
# preceding committed English catalogue. Also reconcile the prior F538 export
# delta (+16/-1): its accepted catalog had 6512 UI rows while this pin still
# recorded 6497. The two deltas are proved separately; no gate is relaxed.
# 2026-10-02 F563: measured canonical inventory after the complete nine-locale
# generator audit. One identity added and none removed relative to the
# preceding committed English catalogue. Translation quality gates unchanged.
# 2026-10-02 F580: measured canonical inventory after the complete nine-locale
# generator audit. Twenty identities added and none removed relative to the
# preceding committed English catalogue. Translation quality gates unchanged.
# 2026-10-02 F572: six reviewed post-save integrity notice identities added,
# none removed; subtracting exactly those keys reproduces the preceding pin.
# 2026-10-03 N615: the alpha batch (556, 558, 560, 564, image QC) catalogued
# with nine-locale AI-reviewed records: +62 UI, -1 UI (the pix2pix prompt
# replaces "Input channels > channel to predict:"), +1 SETTING_LABELS and
# +1 SETTING_TOOLTIPS for counterfactual_condition; then item 631's RAM
# guard: +12 UI and ram_guard's label and tooltip.
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
    "SETTING_LABELS": 1211,
    "SETTING_TOOLTIPS": 1233,
    # 192 -> 201 on 2026-09-08, +9/-0: the nine OPS section headings that
    # fold onto Align & Stitch. Each needed a curated CATEGORY_TOOLTIPS
    # entry or its panel drew the generic fallback -- a heading whose
    # tooltip says nothing about the settings under it, which costs the
    # reader the hover and tells them nothing. 201 -> 200 on 2026-09-11
    # with `save_to_db`, whose help text was one of them.
    "CATEGORY_HELP": 237,
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
    # 2026-10-01: +4 indirect tr() captions: Low/Medium/High from the
    # thumbnail quality selector and Plugins from Preferences._page.
    # All nine source-bound reviewed targets passed the normal builder.
    # 2026-10-01: +5 CellProfiler example export captions; subtracting
    # those exact source keys reproduces the previous 6,489-key inventory.
    # 2026-10-01: CellProfiler provisioning replaces one blurb and adds
    # "Prepare compatible Python": two arrivals, one retirement, net +1.
    # F538: two reviewed opt-in unmixed Measure display captions, no removals.
    # N615 2026-10-03: +62/-1 alpha-batch captions and +12 for 631, named above.
    "UI": 6625,
    "MODULE_SUMMARIES": 72,
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
    # Mask cloud category: Cloud heading (+1 UI), its curated help
    # (+1 UI and +1 CATEGORY_HELP), and no removed identities.
    # 418: 0f21cbfb3's English identities reproduce the previous pin exactly:
    # b4d1896bbc1135f9f4098b3473ac4163d9cf36bf3d58c7447daeb652dacf725f.
    # The +32/-84 identities named above gave 476736243f...a5ac.
    # 316, 2026-09-25: the +2304/-122 identities named over
    # EXTERNAL_SOURCE_COUNTS give this current source digest.
    # 47: one reviewed UI arrival, "Checking compatible GPUs…", no removals.
    # Exact subtraction reproduces the preceding 5a560d33...ef0091b pin.
    # Removing those four indirect UI keys reproduces fa4c72cf...e02433.
    # Subtracting the five example-export UI keys reproduces f4809f01...70741.
    # Removing the two CellProfiler-provisioning arrivals and restoring its
    # retired blurb reproduces e4ee6a9c...c87d3 exactly.
    # Removing the two F538 display captions reproduces fa681c9c...ff8d1d7.
    # N610: replace only the cancellation tooltip with the accurate
    # no-new-suggestions / possible earlier clears-and-scores wording.
    # N615 2026-10-03: the +78/-1 identities (alpha batch and 631) named over
    # EXTERNAL_SOURCE_COUNTS; removing them and restoring the retired
    # prompt reproduces 70748cfb...181c60.
    'd977d86f57aefba253bfda05ac300bdbb306c75ec6a46230c570cdd28dd2da39'
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
    """Return shortcut copy while deliberately excluding key identifiers.

    Copy the GENERATED catalog already owns is left to it. A shortcut named
    after the control it works -- M, "Live magnifier", the Make Masks
    card's own title -- has one caption with one reviewed owner already,
    and counting it here as well would demand a second, exact ``_ROWS`` row
    that `test_compact_and_generated_caption_owners_are_disjoint` then
    refuses: the two tests together made such a shortcut impossible to
    add. That it is translated in every language is asserted by
    `test_every_shortcut_caption_has_exactly_one_translated_owner`.
    """
    from spacr.qt.i18n_catalogs import en
    from spacr.qt.shortcuts import SCREEN_SHORTCUTS, SHORTCUTS

    return {
        value.strip()
        for spec in (*SHORTCUTS, *SCREEN_SHORTCUTS)
        for value in (spec.label, spec.category, spec.scope)
        if value.strip() and value.strip() not in en.UI_SOURCES
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
    from spacr.qt.i18n import _ROWS, _TERM_ROWS
    from spacr.qt.i18n_catalogs import en

    duplicated = sorted((set(_ROWS) | set(_TERM_ROWS)) & set(en.UI_SOURCES))
    assert not duplicated, (
        "compact captions must not acquire a second generated owner:\n  "
        + "\n  ".join(repr(value) for value in duplicated)
    )


def test_every_shortcut_caption_has_exactly_one_translated_owner():
    """Each label, category and scope on the shortcut map is translated.

    By the compact layer's exact row -- which
    `test_compact_user_facing_caption_surface_has_exact_rows_and_is_pinned`
    requires -- or by the generated catalog when the caption is one it
    already owns (see :func:`_shortcut_caption_fields`). This test is the
    second case: a real row in every language, not the English fallback,
    so leaving a shortcut to the generated layer cannot become a way of
    leaving it untranslated.
    """
    from spacr.qt.i18n import _ROWS, VALID_LANGUAGE_CODES, has_translation
    from spacr.qt.i18n_catalogs import en
    from spacr.qt.shortcuts import SCREEN_SHORTCUTS, SHORTCUTS

    copy = {
        value.strip()
        for spec in (*SHORTCUTS, *SCREEN_SHORTCUTS)
        for value in (spec.label, spec.category, spec.scope)
        if value.strip()
    }
    left_to_the_generated_layer = copy - _shortcut_caption_fields()
    assert "Live magnifier" in left_to_the_generated_layer
    for caption in sorted(left_to_the_generated_layer):
        assert caption in en.UI_SOURCES and caption not in _ROWS, caption
        for language in VALID_LANGUAGE_CODES[1:]:
            assert has_translation(caption, language), (caption, language)


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


#: Captions written by a feature branch that the catalog lane has not built
#: rows for yet; spacr/qt/i18n_catalogs is regenerated only by that lane.
#: Owed since 2026-09-25 by item 508 (the enhancement chain reaches Mask
#: generation) and item 509 (PSF optics infer themselves). Self-emptying: a caption already in en.UI_SOURCES fails
#: below and must leave this set.
# 316, 2026-09-25: empty. The catalog pass built rows for every caption
# 508 and 509 left here, and they are in en.UI_SOURCES now.
# 493, 2026-09-26: the parallel GPU mask controls, their greyed-out
# reasons, the Cluster Distribution profile and the per-GPU progress line.
# 569, 2026-09-26: the Show alpha features switch and its tooltip.
# 548, 2026-09-26: the folder-watch progress line on Make Masks.
# 570, 2026-09-26: the Control Charts hit-scoring option (alpha).
# 550, 2026-09-26: the cloud-storage button on the src field of Make Masks
# and Measure, and its browser dialog (alpha).
# 577, 2026-09-26: the Preferences Notifications tab (alpha).
# 572, 2026-09-26: the figure integrity toggle on the Preferences Figures
# tab and its tooltip (alpha).
# 574, 2026-09-27: the Report screen's Archive package button and form.
# 579, 2026-09-27: the Report screen's Deposit on Zenodo button and form.
# 316, 2026-09-27: all 76 resolved through their explicit owners.
# 593, 2026-09-28: Make Masks' Consolidate folders and Sort into channels
# buttons, their prompts and console lines, and the channel-sort dialog,
# regex window and example-sets check (spacr/qt/widgets/channel_sort_dialog.py).
# 316, 2026-09-27 (sixth pass): empty again; the catalog pass built rows
# for every caption left here.
# 600, 2026-09-29: Make Masks' "Organize for Measure…" button and popup
# (spacr/qt/widgets/organize_for_measure.py) and the drop-classification
# console lines and mask-pairing prompt on the Make Masks screen.
# 600b, 2026-09-29: the popup table's Text / Image / Image + text view,
# overlay colour, slot-drag tooltip and the ×'s tooltip.
# 316, 2026-09-30 (eighth pass): empty again; the catalog pass built rows
# for every caption left here.
# 600c, 2026-09-30: the popup's empty-table drop hint and Size slider.
# 316, 2026-09-30 (eighth pass follow-up): 600c's two captions have rows;
# empty again.
# 598, 2026-09-30: the contribute dialogs' dataset-link caption and the
# thank-you caption before the pull request's link.
# 316, 2026-09-30 (tenth pass): 598's two captions have rows; empty again.
# 563, 2026-09-30: the Control Charts anomaly table's Control percentile
# column (alpha), after the real-screen check ranked wells by it.
# 573, 2026-09-30: the analysis lock dialog's Gate files row and its
# placeholder (alpha).
# 422, 2026-09-30: the Help search result-row templates in help_index.py,
# now extractable through `_template`. The list's new "and {count} more" row
# reuses a caption the catalogs already carry, so it owes nothing.
# 585, 2026-09-30: the arrayed-assay planner's readout and condition
# pickers and its plan/load error lines (alpha); "Load plan…" already has
# a row.
# 603, 2026-09-30: Preferences' Tooltip delay row and its explanation.
# 610, 2026-10-01: Annotate's Suggest run -- its step n of N line and step
# names, the Cancel button's tooltip and cancel lines, and the console note
# for a round checked on a random split.
# 2026-10-01: full nine-locale rebuild completed; pending rows are now
# catalogued or retired with their source captions. No translation bypass remains.
# 558, 2026-10-02: the Apply virtual stain button on Mask and Make Masks.
# N615, 2026-10-03: every pending caption above, and item 631's RAM guard
# dialog, catalogued in nine locales; Object and Propagate are compact rows.
# 634, 2026-10-03: the five alpha organism pages (Trypanosoma, Leishmania,
# Giardia, virus, mammalian): their prose, tiles, links, workflow notes,
# UniProt compartment names and SwissBioPics location descriptions.
_AWAITING_CATALOG_REBUILD: frozenset[str] = frozenset({
    # item 633: Load test data on Convert, Layer Viewer and External Masks
    "Download Import's test data, about 285 MB: four microscope fields "
    "with their cell, nucleus and pathogen masks, written in every "
    "format spaCR reads. This screen is filled with the fields it can "
    "show. Cached afterwards.",
    "Opened test data: {path}", "The test data has no images.",
    # item 632: the gate editor's 3D shapes, polygon closing and renaming
    # (the polygon help, renaming and 631's keep button were catalogued by
    # 9d4066a96 and left this list)
    "Polygon through view", "Ellipsoid with handles", "Box with handles",
    # item 634: the five alpha organism pages
    "Amastigote conversion",
    "Amastigote infection",
    "Antiviral response",
    "Attachment and motility",
    "Barrier damage",
    "Biology source: UniProt",
    "Biology source: ViralZone",
    "Cell cycle and compound responses",
    "Cell cycle, uptake and cytotoxicity",
    "Cell migration",
    "Cell migration opens the Motility Assay with infection QC off, because no pathogen is present; it reports track speed and straightness. The planned Wound closure module will measure gap closure in scratch assays. Record the frame interval and confluency, because crowding changes how fast and how straight cells move.",
    "Cell segmentation",
    "Cell segmentation opens Mask to segment nuclei and cells in fixed or live images. Morphology profiling opens Measure to quantify per-cell intensity, texture and shape. The planned Organelle morphology module will measure mitochondrial and other organelle shapes. Choose stains that mark the compartments in the diagram you want to measure, and keep imaging settings fixed across a plate.",
    "Cell segmentation, Morphology profiling, Cell migration, Phagocytosis and Cytotoxicity open existing spaCR modules; the other three tiles are planned and marked Coming soon. UniProt and the Human Protein Atlas give subcellular locations for choosing markers. The diagram shows UniProt vocabulary, not where a selected protein is measured.",
    "Cell-cycle staging",
    "Cell-cycle staging is planned to classify G1, S, G2 and mitotic cells from DNA content and markers. Phagocytosis opens Host–Pathogen Analysis with beads or particles as the pathogen objects. Cytotoxicity opens Dose–Response to fit cell number or a viability readout against compound concentration. Separate internalised from surface-bound particles with a quenching or differential stain before reading uptake, and count cells in each well so that cytotoxicity and proliferation are not confused.",
    "Cell-cycle staging is planned to count kinetoplasts and nuclei per cell, because the kinetoplast divides before the nucleus and the counts mark cell-cycle position. Drug response imaging opens Dose–Response to fit a per-well image readout, such as parasite number, against compound concentration. Report the exposure time with each curve: a delayed cell cycle and parasite killing can both lower the count.",
    "Cellosaurus",
    "Centrosome",
    "Chagas disease: CDC",
    "Cilium",
    "Classify G1, S, G2 and mitotic cells.",
    "Classify life-cycle forms from images.",
    "Classify procyclic and metacyclic promastigotes.",
    "Compound responses and host damage",
    "Count amastigotes per macrophage and infected cells.",
    "Count infected host cells and amastigotes per cell.",
    "Count kinetoplasts and nuclei per cell.",
    "Count trophozoites on epithelial cells.",
    "Count virus-positive host cells per well.",
    "Cultured mammalian cells, from primary cells to established lines, are the hosts in most infection assays and the subject of many screens on their own. Imaging can measure cell number, shape, migration, the cell cycle, organelle organization and responses to compounds, with no pathogen in the well.",
    "Cytopathic effect",
    "Cytotoxicity",
    "Detect fused multinucleated host cells.",
    "Disc morphology",
    "Division and host damage",
    "Drug response imaging opens Dose–Response to fit amastigote clearance or infected-cell fraction against compound concentration. The planned Host cell damage module will measure macrophage loss, so that a compound that kills host cells is not mistaken for one that clears parasites. Count host cells in every well alongside the parasite readout, and include an untreated infected control on every plate so that the curve has a defined top.",
    "Encystation",
    "Encystation and excystation",
    "Encystation is planned to quantify cyst-wall staining and the fraction of trophozoites forming cysts, and Excystation will follow trophozoites emerging from cysts. These transitions take hours and depend on bile, pH and culture conditions, so report the induction protocol and the time of imaging with the result. Cysts are small and refractile, so a wall stain and a fixed focal plane help separate them from debris.",
    "Endosome",
    "Endosomes are highly dynamic membrane systems involved in transport within the cell, they receive endocytosed cell membrane molecules and sort them for either degradation or recycling back to the cell surface. They also receive newly synthesised proteins destined for vacuolar/lysosomal compartments. In certain cell types, endosomal multivesicular bodies may fuse with the cell surface in an exocytic manner. These released vesicles are called exosomes.",
    "Entry, fusion and antivirals",
    "Epithelial attachment",
    "Epithelial attachment opens Host–Pathogen Analysis to count trophozoites on epithelial host cells; a single plane cannot separate attachment from overlap, so confirm it with a focal plane at the cell surface. Trophozoite motility opens the Motility Assay with infection QC off to measure swimming speed and straightness. The planned Disc morphology module will measure the ventral disc.",
    "Epithelial attachment, Trophozoite motility and Drug response imaging open existing spaCR modules; the other five tiles are planned and marked Coming soon. GiardiaDB provides genome context and UniProt supplies protein annotations. The generic cell shows UniProt vocabulary, not Giardia anatomy or the measured location of a protein.",
    "Excystation",
    "Extracellular matrix",
    "Flagellar motility",
    "Flagellar motility opens the Motility Assay with infection QC off, because swimming trypanosomes have no host cell; it reports track speed and straightness. The planned Stage differentiation module will classify bloodstream, stumpy and procyclic forms of T. brucei, or trypomastigotes, epimastigotes and amastigotes of T. cruzi. Record the species and life-cycle form, because flagellum length and swimming behaviour differ between them.",
    "Flagellar motility, Host-cell invasion, Amastigote infection and Drug response imaging open existing spaCR modules; the other four tiles are planned and marked Coming soon. TriTrypDB provides genome and gene context and UniProt supplies protein annotations. The highlighted compartments are UniProt vocabulary for choosing markers, not a measured location for a selected gene.",
    "Flagellum",
    "Flagellum axoneme",
    "Flagellum basal body",
    "Follow both nuclei through mitosis.",
    "Follow infection foci expanding over time.",
    "Follow parasite release from host cells.",
    "Follow promastigote-to-amastigote transformation.",
    "Follow trophozoite emergence from cysts.",
    "Giardia duodenalis",
    "Giardia duodenalis, also called G. lamblia or G. intestinalis, causes giardiasis, a diarrhoeal disease of the small intestine. Trophozoites swim with four pairs of flagella, carry two nuclei and attach to the intestinal epithelium with a ventral disc; infection spreads through environmentally resistant cysts. Imaging can measure attachment, motility and encystation.",
    "GiardiaDB",
    "Glycosome",
    "Host Golgi apparatus",
    "Host cell membrane",
    "Host cytoplasm",
    "Host cytoskeleton",
    "Host endoplasmic reticulum",
    "Host endosome",
    "Host endosomes are highly dynamic membrane systems involved in transport within the host cell, they receive endocytosed host cell membrane molecules and sort them for either degradation or recycling back to the host cell surface. They also receive newly synthesised proteins destined for host vacuolar/lysosomal compartments.",
    "Host mitochondrion",
    "Host nucleolus",
    "Host nucleus",
    "Host perinuclear region",
    "Host-cell infection by Trypanosoma cruzi",
    "Host-cell invasion",
    "Host-cell invasion opens the Invasion Assay, which separates attached from internalised trypomastigotes when two-colour differential staining is available. Amastigote infection opens Host–Pathogen Analysis to count infected host cells and amastigotes per cell. The planned Trypomastigote egress module will follow parasite release, and Host cell damage will describe monolayer loss. These readouts apply to T. cruzi; T. brucei does not invade cells.",
    "Human Protein Atlas",
    "ICTV virus taxonomy",
    "Image-analysis modules for Giardia duodenalis",
    "Image-analysis modules for Leishmania parasites",
    "Image-analysis modules for Trypanosoma brucei and Trypanosoma cruzi",
    "Image-analysis modules for mammalian host cells without a pathogen",
    "Image-analysis modules for virus-infected host cells",
    "Infection and replication sites",
    "Infection rate",
    "Infection rate opens Host–Pathogen Analysis to count host cells positive for a viral antigen or reporter among all host cells. Recruitment opens the Recruitment module to measure host-protein enrichment at viral replication compartments. Define the positivity threshold from mock-infected wells, and record the multiplicity of infection and fixation time with every measurement.",
    "Infection rate, Plaque Assay, Recruitment and Antiviral response open existing spaCR modules; the other four tiles are planned and marked Coming soon. ViralZone describes virus families and their replication cycles, and UniProt supplies viral and host protein annotations. The diagram shows UniProt vocabulary, not where a selected protein is measured.",
    "Kinetoplast",
    "Leishmania parasites cause cutaneous, mucosal and visceral leishmaniasis and are transmitted by sand flies. Flagellated promastigotes develop in the insect, and amastigotes replicate inside macrophages in a parasitophorous vacuole. Imaging can measure macrophage infection, parasite load, promastigote motility, stage conversion and responses to compounds.",
    "Leishmania spp.",
    "Lysosome",
    "Macrophage binding",
    "Macrophage infection",
    "Macrophage infection and parasite load",
    "Macrophage infection opens Host–Pathogen Analysis to report the fraction of infected macrophages and the amastigotes per macrophage, the two usual measures of parasite load. Macrophage binding opens the Invasion Assay to separate bound from internalised promastigotes when differential staining is used. The planned Vacuole size module will measure parasitophorous vacuole area, which differs between Leishmania species and grows as amastigotes multiply inside it.",
    "Macrophage infection, Macrophage binding, Promastigote motility and Drug response imaging open existing spaCR modules; the other four tiles are planned and marked Coming soon. TriTrypDB provides genome context and UniProt supplies protein annotations. The diagram is shared with Trypanosoma and shows UniProt vocabulary, not the measured location of a Leishmania protein.",
    "Mammalian cells",
    "Measure amastigote clearance across compound concentrations.",
    "Measure cell number across compound concentrations.",
    "Measure epithelial monolayer integrity.",
    "Measure gap closure in scratch assays.",
    "Measure host-cell rounding, detachment and loss.",
    "Measure host-protein enrichment at replication sites.",
    "Measure infection across antiviral concentrations.",
    "Measure macrophage loss and monolayer integrity.",
    "Measure mitochondrial and organelle shape.",
    "Measure parasite growth across compound concentrations.",
    "Measure parasitophorous vacuole area per infected cell.",
    "Measure per-cell intensity and shape features.",
    "Measure trophozoite growth across compound concentrations.",
    "Measure ventral disc shape and integrity.",
    "Metacyclogenesis",
    "Microtubule organizing center",
    "Migration and wound closure",
    "Morphology profiling",
    "Motility and life-cycle forms",
    "NCBI Virus",
    "Nuclear division",
    "Nuclear division is planned to follow the two nuclei through mitosis, which a cell-cycle readout must count as a pair. Barrier damage will measure epithelial monolayer integrity after exposure to trophozoites. Drug response imaging opens Dose–Response to fit trophozoite number or attachment against compound concentration. Detached trophozoites are lost when wells are washed, so decide whether attachment or survival is the endpoint.",
    "Nucleus speckle",
    "Opens Dose–Response. Fit a four-parameter logistic curve and EC50 to a per-well image readout, such as parasite number, against compound concentration.",
    "Opens Dose–Response. Fit a four-parameter logistic curve and EC50 to cell number or viability per well against compound concentration.",
    "Opens Dose–Response. Fit a four-parameter logistic curve and EC50 to the infected fraction or amastigotes per macrophage against compound concentration.",
    "Opens Dose–Response. Fit a four-parameter logistic curve and EC50 to the infected fraction per well against antiviral concentration.",
    "Opens Dose–Response. Fit a four-parameter logistic curve and EC50 to trophozoite number or attachment per well against compound concentration.",
    "Opens Host–Pathogen Analysis. Measure all host cells and viral antigen or reporter signal as pathogen objects; its infected fraction per well is the infection rate.",
    "Opens Host–Pathogen Analysis. Measure epithelial cells as host cells and trophozoites as pathogen objects; it reports the fraction of host cells with trophozoites and trophozoites per cell.",
    "Opens Host–Pathogen Analysis. Measure host cells, including uninfected ones, and amastigotes as pathogen objects; it reports the infected fraction and amastigotes per cell.",
    "Opens Host–Pathogen Analysis. Measure macrophages as host cells and amastigotes as pathogen objects; it reports the infected fraction and amastigotes per macrophage.",
    "Opens Host–Pathogen Analysis. Measure the cells as host cells and beads or particles as pathogen objects; it reports the fraction of cells with uptake and particles per cell.",
    "Opens Mask. Choose a Cellpose model for nuclei and cells; it writes masks for Measure and the other modules.",
    "Opens Measure. Point it at the images and masks; it writes per-cell intensity, texture and shape features.",
    "Opens Recruitment. Use the viral replication compartment as the pathogen object; it reports host-protein enrichment around it.",
    "Opens the Invasion Assay. Two-colour differential staining separates attached from internalised T. cruzi trypomastigotes, as for Toxoplasma invasion.",
    "Opens the Invasion Assay. Two-colour differential staining separates bound from internalised promastigotes.",
    "Opens the Motility Assay with infection QC off, because no pathogen is present. Segment the cells as the tracked objects; it reports track speed and straightness per well.",
    "Opens the Motility Assay with infection QC off, because promastigotes in culture have no host cell. Segment them as the tracked cell objects; it reports track speed and straightness per well.",
    "Opens the Motility Assay with infection QC off, because swimming trophozoites have no host cell. Segment them as the tracked cell objects; it reports track speed and straightness per well.",
    "Opens the Motility Assay with infection QC off, because swimming trypanosomes have no host cell. Segment the parasites as the tracked cell objects; it reports track speed and straightness per well.",
    "Opens the Plaque Assay. Segment plaques in the stained monolayer; it reports plaque number and size per well.",
    "Organelle morphology",
    "Plaque Assay opens the Plaque Assay module to count and measure plaques in a stained monolayer. The planned Viral spread module will follow infection foci over time, and Cytopathic effect will measure host-cell rounding, detachment and loss. Plaque size integrates several rounds of replication and spread, so compare it with the infection rate before assigning a defect to one step.",
    "Plaques and spread",
    "Promastigote development and motility",
    "Promastigote motility",
    "Promastigote motility opens the Motility Assay with infection QC off to measure swimming speed and straightness. The planned Metacyclogenesis module will separate procyclic from infective metacyclic promastigotes by shape, and Amastigote conversion will follow the transformation into rounded amastigotes. Record the culture day and medium, because promastigote populations change composition as cultures age.",
    "Quantify cyst-wall staining and cyst formation.",
    "Quantify uptake of beads or particles.",
    "Segment nuclei and cells in fixed or live images.",
    "Segmentation and morphology",
    "Separate attached from internalised trypomastigotes.",
    "Separate bound from internalised promastigotes.",
    "Separate bound from internalised virions.",
    "Stage differentiation",
    "SwissBioPics has no Giardia drawing, so this is its generic eukaryotic cell. Hover for a UniProt location description and check several compartments to keep them highlighted. The labels are UniProt locations annotated for Giardia proteins and drawn here. Giardia has two nuclei, mitosomes instead of mitochondria and no stacked Golgi, so those are not offered.",
    "Syncytium formation",
    "The SwissBioPics animal cell. Hover for a UniProt location description and check several compartments to keep them highlighted. The labels are UniProt subcellular locations annotated for mammalian proteins that this artwork draws. Cell shape and organelle arrangement vary between cell types.",
    "The SwissBioPics host animal cell with a virion. Hover for a UniProt location description and check several compartments to keep them highlighted. The labels are the virion and host-cell locations UniProt annotates for viral proteins that this artwork draws. It is generic: virion structure and replication sites differ between virus families.",
    "The SwissBioPics trypanosomatid cell, shared with the Trypanosoma page. Hover for a UniProt location description and check several compartments to keep them highlighted. The labels are the UniProt subcellular locations annotated for Leishmania proteins that this artwork draws. Promastigotes and the rounded amastigote differ in shape and flagellum length.",
    "The SwissBioPics trypanosomatid cell. Hover for a UniProt location description and check several compartments to keep them highlighted. The labels are the UniProt subcellular locations annotated for Trypanosoma proteins that this artwork draws. It shows one trypomastigote-like form; amastigotes and epimastigotes differ in flagellum length and organelle position.",
    "The basal body is a barrel-shaped microtubule-based structure required for the formation of flagella. Basal bodies, structuraly related to and often interconvertible with centrioles, serves as a nucleation site for axoneme growth.",
    "The centrosome is a microtubule organizing center (MTOC) responsible for the nucleation and organisation of  microtubules. It is composed of two orthogonally arranged centrioles, each one having a barrel shaped microtubule structure, and their surrounding pericentriolar material (PCM).",
    "The cilium is a cell surface projection found at the surface of a large proportion of eukaryotic cells. The two basic types of cilia, motile (alternatively named flagella) and non-motile, collectively perform a wide variety of functions broadly encompassing cell/fluid movement and sensory perception. Their most prominent structural component is the axoneme which consists of nine doublet microtubules, with all motile cilia - except those at the embryonic node - containing an additional central pair of microtubules. The axonemal microtubules of all cilia nucleate and extend from a basal body, a centriolar structure most often composed of a radial array of nine triplet microtubules. In most cells, basal bodies associate with cell membranes and cilia are assembled as 'extracellular' membrane-enclosed compartments.",
    "The extracellular matrix (ECM) is a vague term used to refer to all the material surrounding cells in a multicellular organism, except circulating fluids such as blood or lymph. In some cases, the ECM accounts for more of the organism's bulk than its cells. In plants, arthropods and fungi the ECM is primarily composed of nonliving material such as cellulose or chitin. In vertebrates the ECM consists of a complex network including the basement membrane, collage, elastin, proteoglycans and hyaluronan.",
    "The flagellum axoneme is the most prominent structural component of the flagellum, which is a long whip-like or feathery structure which propels the cell through a liquid medium. The flagellum axoneme consists of a characteristic axial '9+2' microtubular array.",
    "The flagellum is a long whip-like or feathery structure which propels the cell through a liquid medium. This motile cilium is produced by the unicellular eukaryotes, and by the motile male gametes of many eukaryotic organisms. The flagella commonly have a characteristic axial '9+2' microtubular array (axoneme) and bends are generated along the length of the flagellum by restricted sliding of the nine outer doublets.",
    "The glycosome is a specialized peroxisome found in all members of the protist order Kinetoplastida examined. Nine enzymes involved in glucose and glycerol metabolism are associated with these organelles. These enzymes are involved in pathways which, in other organisms, are usually located in the cytosol.",
    "The host Golgi apparatus is a series of flattened, cisternal membranes and similar vesicles usually arranged in close apposition to each other to form stacks. In mammalian cells, the host Golgi apparatus is juxtanuclear, often pericentriolar. The stacks are connected laterally by tubules to create a perinuclear ribbon structure, the 'Golgi ribbon'. In plants and lower animal cells, the host Golgi exists as many copies of discrete stacks dispersed throughout the host cytoplasm. It is a polarized structure with, in most higher eukaryotic cells, a cis-face associated with a tubular reticular network of membranes facing the endoplasmic reticulum, the cis-Golgi network (CGN), a medial area of disk-shaped flattened cisternae, and a trans-face associated with another tubular reticular membrane network, the trans-Golgi network (TGN) directed toward the host plasma membrane and compartments of the host endocytic pathway.",
    "The host cell membrane is the selectively permeable membrane which separates the host cytoplasm from its surroundings. Known as the host cell inner membrane in prokaryotes with 2 membranes.",
    "The host cytoplasm is the content of a host cell within the plasma membrane and, in eukaryotics cells, surrounds the host nucleus.",
    "The host cytoskeleton is a dynamic three-dimensional structure that fills the host cytoplasm of eukaryotic cells. It is responsible for cell movement, cytokinesis, and the organization of the organelles or organelle-like structures within the host cell.",
    "The host endoplasmic reticulum (ER) is an extensive network of membrane tubules, vesicles and flattened cisternae (sac-like structures) found throughout the eukaryotic host cell, especially those responsible for the production of hormones and other secretory products.",
    "The host mitochondrion is a semiautonomous, self-reproducing organelle that occurs in the cytoplasm of all cells of most, but not all, host eukaryotes. Each host mitochondrion is surrounded by a double limiting membrane. The inner membrane is highly invaginated, and its projections are called cristae. They are the sites of the reactions of oxidative phosphorylation, which result in the formation of ATP.",
    "The host nucleolus is a dark, dense, roughly spherical area of fibers and granules in the host nucleus. Only plant and animal nuclei contain one or more nucleoli, although some do not. No membrane separates the host nucleolus from the host nucleoplasm. It mediates ribosomal RNA biogenesis.",
    "The host nucleus is the most obvious organelle in any host eukaryotic cell. It is a membrane-bound organelle and is surrounded by double membranes. It communicates with the surrounding cytosol via numerous nuclear pores.",
    "The host perinuclear region is the host cytoplasmic region just around the host nucleus.",
    "The lysosome is a membrane-limited organelle present in all eukaryotic cells, which contains a large number of hydrolytic enzymes that are used for degrading almost any kind of cellular constituent, including entire organelles. The mechanisms responsible for delivering cytoplasmic cargo to the lysosome/vacuole are known collectively as autophagy and play an important role in the maintenance of homeostasis.",
    "The membrane surrounding the virion.",
    "The microtubule organizing center (MTOC) is an intracellular structure that can catalyze gamma-tubulin-dependent microtubule nucleation and that can anchor microtubules.",
    "The mitochondrial DNA of trypanosomatid protozoa is termed kinetoplast DNA (kDNA). kDNA is a massive network, composed of thousands of topologically interlocked DNA circles. Each cell contains one network condensed into a disk-shaped structure within the matrix of its single mitochondrion. The kDNA circles are of two types, maxicircles present in a few dozen copies and minicircles present in several thousand copies.",
    "The nuclear speckles are small subnuclear membraneless organelles or structures, also called the splicing factor (SF) compartments that correspond to nuclear domains located in interchromatin regions of the nucleoplasm of mammalian cells. Protein found in speckles serves as a reservoir of factors that participate in transcription and pre-mRNA processing. Speckles appear, at the immunofluorescence-microscope level, as irregular, punctuate structures, which vary in size and shape. Usually 25-50 speckles are observed per interphase mammalian nucleus. At the electronic-microscope level, they are composed of heterogeneous mixture of electro-dense particles with diameters ranging from 20-25 nm and are called interchromatin granules clusters (IGCs). Speckles are dynamic structures. Both their protein and RNA-protein components can cycle continuously between speckles and other nuclear locations depending on the transcriptional state of the cell. Structures similar to nuclear speckles have been identified in the amphibian oocyte nucleus (called B snurposomes) and in Drosophila melanogaster embryos, but not in yeast.",
    "The viral tegument is a protein structure that resides between the capsid and envelope of herpesviruses and which appears amorphous in electron micrographs.",
    "The virion is the complete fully infectious extracellular virus particle.",
    "Track cell speed and straightness over time.",
    "Track swimming speed and straightness.",
    "TriTrypDB",
    "Trophozoite motility",
    "Trypanosoma brucei causes African sleeping sickness and lives outside host cells in blood and tissue fluids, while Trypanosoma cruzi causes Chagas disease and replicates inside host cells as amastigotes. Both are flagellated kinetoplastids whose single mitochondrion carries a kinetoplast. Imaging can follow motility, host-cell infection, the cell cycle and drug responses.",
    "Trypanosoma spp.",
    "Trypomastigote egress",
    "Vacuole size",
    "Viral spread",
    "ViralZone",
    "Virion",
    "Virion membrane",
    "Virion tegument",
    "Virus entry",
    "Virus entry is planned to separate bound from internalised virions, and Syncytium formation will detect fused, multinucleated host cells. Antiviral response opens Dose–Response to fit the infected fraction against compound concentration. Count host cells in each well too, so that a cytotoxic compound is not read as an antiviral, and keep the multiplicity of infection constant across the dilution series.",
    "Virus infection",
    "Viruses replicate only inside host cells, using host machinery in compartments that differ between virus families: many DNA viruses replicate in the nucleus, and many RNA viruses build replication organelles in the cytoplasm. Imaging can measure the infected fraction of cells, plaques, the spread of infection and responses to antiviral compounds.",
    "Wound closure",
})



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
    assert not _AWAITING_CATALOG_REBUILD & set(en.UI_SOURCES)
    candidates -= _AWAITING_CATALOG_REBUILD
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

    from spacr.qt.i18n import _ROWS, _TERM_ROWS
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
    assert not ((set(_ROWS) | set(_TERM_ROWS)) & set(external["UI"])), (
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

    from spacr.qt.i18n import _ROWS, _TERM_ROWS
    from spacr.qt.i18n_catalogs import CATALOG_LANGUAGES, en

    discovered = set(builder.extract_static_ui_sources())
    independently_expected = (
        _custom_widget_literal_captions() | _indirect_registry_captions()
    ) - _AWAITING_CATALOG_REBUILD
    compact = set(_ROWS) | set(_TERM_ROWS)
    missing_ownership = sorted(
        independently_expected
        - _RUNTIME_IDENTITY_CAPTIONS
        - compact
        - discovered
    )
    assert not missing_ownership, (
        "custom/indirect runtime captions bypass both i18n layers:\n  "
        + "\n  ".join(repr(value) for value in missing_ownership)
    )

    ambiguous_ownership = sorted(
        source
        for source in independently_expected - _RUNTIME_IDENTITY_CAPTIONS
        if int(source in compact) + int(source in en.UI_SOURCES) != 1
    )
    assert not ambiguous_ownership, (
        "custom/indirect captions need exactly one compact or generated "
        "owner:\n  "
        + "\n  ".join(repr(value) for value in ambiguous_ownership)
    )

    external = (
        independently_expected - _RUNTIME_IDENTITY_CAPTIONS - compact
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
