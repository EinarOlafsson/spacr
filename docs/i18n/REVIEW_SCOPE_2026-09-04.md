# Localization review scope at the 1.5.0.5 release gate — 2026-09-04

This supersedes `REVIEW_SCOPE_2026-08-30.md`, which is kept as history. Like
that report, **this is an evidence report and not a certificate that every
sentence was read by a fluent speaker.** The semantic-review evidence is
defect-driven and much smaller than the shipped corpus.

**Mechanical source coverage IS complete, as of 2026-09-10.** It was not on
2026-09-06, and this paragraph carried that gap until now: the runtime
catalogs held 5,105 of 5,226 entries in every locale -- 121 sources missing
each, 1,089 in total -- and the API catalogs 8,966 of 10,241. The machine
that measured it had no OPUS checkpoint, so the gap was recorded rather than
closed.

It is closed. Both audits pass on the same tree:

    verified API catalogs: languages=9 symbols=10306
    verified external runtime catalogs: languages=9 settings=1073
      categories=201 ui=2841 modules=67

and `docs/i18n/COVERAGE.md`, regenerated beside this file, reports
`5260/5260` runtime and `10296/10296` API for every one of the nine, with
zero orphans. Two hand-written records were needed where the model could not
satisfy the gates: `spacr.ops_accel#4` in Portuguese and the `ops_gpu`
tooltip in Simplified Chinese.

WHAT REMAINS IS THE REVIEW, WHICH IS THE POINT OF THIS FILE. Mechanical
coverage means every source has A translation. It does not mean a speaker of
the language has read it, and the table below is what says how much of that
has actually happened -- between 1.6 % and 9.9 % per locale.

## What changed since 2026-08-30

The runtime inventory grew from 4,982 entries to **5,165**: 1,019 setting
labels, 1,014 setting tooltips, 192 category explanations, 2,817 UI strings, 57
installer strings and 66 module summaries. The API inventory grew from 8,861
symbol documents to **9,434**, all ten locales in agreement.

Every mechanical gate now passes. `tests/qt/test_external_i18n_catalogs.py` is
31 of 31, having been 7 failed / 24 passed on the morning of 2026-09-04.

## Reviewed evidence, per locale

Source-bound records under `docs/i18n/reviewed/runtime/<locale>/` and
`docs/i18n/reviewed/api/<locale>/`, against the LIVE denominators rather than a
remembered one: 5,993 runtime entries and 10,931 public API docstrings. As before,
repeated source strings mean this is not a unique-string percentage, and the
proportion is small by design: the evidence is defect-driven.

Both denominators move whenever a string or a docstring is added, so these
numbers are regenerated rather than transcribed; the test that guards this
table derives them from the same source the builders read.

| Language | Reviewed runtime records | Of 5,993 | Remainder | Reviewed API blocks | Of 10,931 | Remainder |
|---|---:|---:|---:|---:|---:|---:|
| Swedish | 319 | 5.32% | 5,674 | 710 | 6.50% | 10,221 |
| German | 282 | 4.71% | 5,711 | 691 | 6.32% | 10,240 |
| Spanish | 284 | 4.74% | 5,709 | 570 | 5.21% | 10,361 |
| Simplified Chinese | 596 | 9.94% | 5,397 | 853 | 7.80% | 10,078 |
| Portuguese | 294 | 4.91% | 5,699 | 798 | 7.30% | 10,133 |
| Hindi | 383 | 6.39% | 5,610 | 747 | 6.83% | 10,184 |
| Korean | 502 | 8.38% | 5,491 | 820 | 7.50% | 10,111 |
| Icelandic | 451 | 7.53% | 5,542 | 1,316 | 12.04% | 9,615 |
| French | 310 | 5.17% | 5,683 | 734 | 6.71% | 10,197 |

*Re-measured 2026-09-16 for 317 using the actual live source extractors and
reviewed-record loaders. Runtime sources increase 5,767 -> 5,773 (seven
example-pack notices added, one superseded notice removed); each locale
adds seven accepted runtime records. The API symbol count stays at 10,539,
with six reviewed blocks per locale for the changed example-settings
importer. All 117 new records explicitly identify AI semantic review, not
native-speaker approval. The historical API block-count/symbol-count ratio
below is not the fraction of complete pages reviewed.*

*Re-measured 2026-09-16 for 418 using the actual source extractors and
reviewed-record loaders, against the clean `f7df13e93` worktree. Runtime
sources decrease 5,819 -> 5,767: labels -21, tooltips -21, categories -1,
UI -9, module summaries unchanged. The previous report's 5,804 denominator
had already missed earlier work; it is not the baseline for this feature.
Accepted runtime evidence increases 3,188 -> 3,504 after source retirement
and source-string deduplication. The API inventory stays at 10,539 symbols;
285 new API records minus two genuinely stale retired records increase
accepted API blocks 7,067 -> 7,350. All new evidence explicitly identifies
AI-assisted semantic review, not native-speaker approval. API percentages
retain this report's historical block-count/symbol-count denominator and
must not be read as the fraction of complete API pages reviewed.*

*Regenerated 2026-09-15 on nightly 08a2c1719, on top of runtime pass A, for
the API lane pass that lands items 412, 416 and 413 and catalogues the work
session's seven symbols, and for runtime pass B on the same branch
(wip/api-pass-412-416-413), with the test's own derivation. The API
denominator moves 10,523 -> 10,539, +16/-0: `spacr.qt.make_masks_demo` and its
seven public functions (412), and `spacr.install_cleanup` with its seven public
symbols (416); 413 rewrote two existing docstrings and added none, and the
work session's `SearchThresholds` pair was already in 10,523. Every reviewed
API count rises by exactly 101, all of it
`docs/i18n/reviewed/api/<locale>/2026-09-15-api-pass-412-416-413.json`: 88
records for 412, 416 and 413, and 13 for the new or edited blocks of the work
session's seven symbols, none of whose sources is in any other record.
The runtime denominator moves 5,769 -> 5,790, +21/-0: 412's tooltip and five
status strings and 416's fifteen update-dialog strings, which are exactly the
live-minus-nightly source set. Every runtime count rises by those 21
(`412-make-masks-demo.json` and `2026-09-15-update-removes-old-installs.json`,
sources in no other file), and Hindi, Korean and Icelandic by one more, the
reviewed "Apply" record in `2026-09-15-apply-terms.json`, whose source was
already counted. So every Remainder is unchanged except those three, which
fall by one. Totals: 2,834 + 9 x 21 + 3 = 3,026.*

*Regenerated 2026-09-15 for runtime pass A on nightly 5a9c4563a, which
carries 387's Dose-Response work, on top of the pass below. The runtime
denominator moves 5,747 -> 5,769: +21 UI for 387's Dose-Response captions,
which nightly added without this report, and +1 for this pass -- UI +4/-3
(Mask Generation's OPS switch tooltip, catalogued for the first time, and the
three rewritten OPS category explanations replacing their mosaic-era wording)
and CATEGORY_HELP +3/-3 for the same three. Reviewed runtime records, from
nightly's live counts (sv 225, de 189, es 198, zh_CN 488, pt 186, hi 269,
ko 390, is 335, fr 203; 2,483, which is 407's 2,429 plus the work session's
six in each locale): +131 Dose-Response terms and Cache ceiling
(cdbb41cbd), +99 for the new OPS English and "Controls", +133 for the fifteen
Dose-Response grid captions the work session left unrecorded (fr and pt
"Doses" are MANUAL_UI identity rows, not records), and -12 retired OPS
records whose English is gone. sv 263, de 226, es 232, zh_CN 531, pt 222,
hi 311, ko 432, is 379, fr 238: 2,483 + 363 - 12 = 2,834. The API side does
not move for this pass: no public symbol changed here. Its denominator
reads 10,523, not 10,521, because nightly's own public surface grew by two
(d57d0d657, "Public surface back to +2"); every reviewed API count is
unchanged and only the percentages and remainders follow.*

*Regenerated 2026-09-15 for the magnifier's border option and whole-image
mode (407), rebased onto nightly df1216b3f, on top of the pass below. The
runtime denominator moves 5,729 -> 5,747, +21/-3 and all UI: 21 new captions
(the border checkbox and its tooltip, the Segment row with its two choices and
their tooltip, a reworded card subtitle, Size tooltip and Magnifier-button
tooltip, and eleven status lines) and the three earlier wordings they replace.
EVERY RUNTIME COUNT RISES BY 18 AND NO REMAINDER MOVES: each locale gains the
21 hand-written records of 2026-09-15-magnifier-whole-image.json and loses the
3 records for the retired wordings, deleted from
2026-09-15-live-magnifier.json. 2,267 + 9 x 18 = 2,429. The API side does not
move; no public symbol changed.*

*Regenerated 2026-09-15 for the deletion of the old OPS engine (372), on top of
the pass below. The API denominator moves 10,541 -> 10,521, +0/-20: the 17
symbols of the old engine's module and the three stitcher defaults in
`spacr.settings` (`set_default_stitch`, `set_default_multichannel`,
`set_default_general`) that nothing called. The runtime denominator moves
5,845 -> 5,729, -116/+0: the labels, tooltips and six category explanations
of the settings only the old engine read. Reviewed runtime records lose the
73 the evidence checker reported stale (sv 1, pt 3, hi 2, ko 18, is 24,
zh_CN 24, fr 1) and gain one per locale for `recursive`, whose English now
comes from `spacr.external_masks` and whose cached translations were wrong in
several locales. Reviewed API blocks lose the records of the old engine's
deleted symbols and the French one for the `spacr.ops_settings` paragraph
that described it, and every locale gains one for the new `ops_defaults`
summary.*

*Regenerated 2026-09-15 for instruction 372's OPS engine, on local nightly
f5d651454 and on top of the pass below. The API denominator moves 10,534 ->
10,541, +7/-0: spacr.ops_engine and run_ops, ops_cycles.AlignedField and
align_field, ops_sbs.attribute_reads and assign_reads_to_objects,
ops_phenotype.phenotype_centres. Every reviewed API count rises by exactly 77:
78 hand-written records per locale (702), two of which share one English
source. The runtime denominator and every runtime count are unchanged. The
totals sentence below is re-derived too: nightly's still read 2,241, the count
before 286's ten records per locale, while its rows already read 2,331.*

*Regenerated 2026-09-15, fourth pass, on nightly b03e9740b, for 286's
performance levels. The runtime denominator moved 5,835 -> 5,845: ten `ui`
captions, the five level notes and the five hardware labels, which 89094e3c4
catalogued together with a reviewed record for each in all nine locales
(`docs/i18n/reviewed/runtime/<locale>/2026-09-15-performance-levels.json`).
So EVERY REVIEWED RUNTIME COUNT IS THE THIRD PASS'S PLUS TEN: sv 191 -> 201,
de 154 -> 164, es 163 -> 173, zh_CN 477 -> 487, pt 154 -> 164, hi 236 -> 246,
ko 373 -> 383, is 324 -> 334, fr 169 -> 179 (2,241 -> 2,331 in all), and every
runtime remainder is unchanged. No public symbol and no reviewed API record
moved, so the API columns are the third pass's as they stood.*

*Regenerated 2026-09-15, third pass, when local nightly (91ac88b51) was
rebased onto the work session's batch (5fce82971). Neither parent's rows hold
alone. The runtime denominator and every runtime count are local nightly's
(5,835 and 2,241 in all), because the side branch moved no caption and no
reviewed runtime record; the API denominator is the side branch's 10,534,
because local nightly moved no public symbol. EVERY REVIEWED API COUNT IS
LOCAL NIGHTLY'S PLUS EIGHT, the eight blocks of `mark_to_start_on`'s record:
sv 531 -> 539, de 489 -> 497, es 370 -> 378, zh_CN 640 -> 648, pt 601 -> 609,
hi 534 -> 542, ko 607 -> 615, is 1,102 -> 1,110, fr 530 -> 538. Proved by
subtraction from both parents with the test's own derivation: dropping that
one key from `public_docstrings()` returns local nightly's committed row in
every locale, and the side branch's committed row plus local nightly's own
rise over b740c741d (sv +58, de +59, es +79, zh_CN +119, pt +149, hi +68,
ko +149, is +71, fr +60) gives the same number, so the two batches' records
share no English source.*

*Regenerated 2026-09-15 for the 08:15 integration batch (wip/integ-0815) on
nightly dad943037, on top of 366's pass below. The runtime denominator moves 5,679 -> 5,835, +156/-0: 154
UI captions that 394's keyed extractor rules found behind local helpers
(always on screen, never in a catalog), plus the `segmentation_backend` label
and tooltip (404/405). The API denominator does not move; the new backend
module is not rendered. EVERY RUNTIME COUNT RISES, and each rise is proved by
subtraction from nightly's count. It adds 2 for `segmentation_backend`, then
the records from a read of every row the batch's GPU pass changed: the
reviewers' corrections as written, plus the ones the gates had refused,
repaired to keep a quoted literal, a code literal's order or an ** span, and
the three UI rows the model left English (sv 55, de 51, es 46, zh_CN 107,
pt 37, hi 86, ko 106, is 149, fr 51). That is 1,535 + 18 + 688 = 2,241, and no
record left any locale.

The API counts rise over 366's by the same read (sv 36, de 37, es 57, zh_CN 96,
pt 127, hi 46, ko 126, is 52, fr 38), plus `spacr.report#13` in zh_CN, ko and
is.
"Segmentation QC" is also a UI caption, and the API block now carries the
runtime reviewers' translation. Icelandic adds 49 sources, not 53, for two
reasons. Two of its corrections are the same "Design notes:" block in two
symbols, one English source with one translation. And three replace the
translations of 2026-09-15-rebound-63 records, retired in place, for sources
already counted. The ko UI row "Choose a regression results folder" keeps the
batch's own record, which matches its sibling rows. Coverage is not review:
these are corrections to text that was wrong, and the Icelandic and Hindi
ones still want a native reader.*

*Regenerated 2026-09-15 for instruction 366's module landing pages, on top of
407's pass below. Neither denominator moves: six existing docstrings grew
paragraphs and no symbol arrived. Every reviewed API count rises by exactly
22, the landing pages' new blocks, each carried by a hand-written record in
all nine locales (198). No runtime count moves.*

*Regenerated 2026-09-15, second pass, for `spacr.graph_types.mark_to_start_on`
(293). The API denominator moves 10,533 -> 10,534, +1/-0 by set difference:
that one function arrives and nothing leaves. Its sibling in `spacr.ml` went
private (`_qc_graph_type_and_note`) before the catalogs were built, so it never
entered the surface. The runtime denominator does not move and no reviewed
runtime record changed, so the runtime columns and the 1,535 total are what
they were. EVERY REVIEWED API COUNT RISES BY EIGHT, the eight blocks of the new
symbol's reviewed record in each locale: sv 473 -> 481, de 430 -> 438, es 291
-> 299, zh_CN 521 -> 529, pt 452 -> 460, hi 466 -> 474, ko 458 -> 466, is
1,031 -> 1,039, fr 470 -> 478. Proved by subtraction: the same derivation with
that one key dropped from `public_docstrings()` returns 10,533 and the previous
row for every locale, number for number.*

*Regenerated 2026-09-15 for instruction 407's live magnifier in Make Masks.
The runtime denominator moves 5,662 -> 5,679, +17/-0 and all UI: the tool-row
button, the card title, the Classical/Cellpose and Clip/Replace choices,
'Updating…', four status lines and six tooltips. The API denominator does not
move; no public symbol and no module arrived. EVERY REVIEWED RUNTIME COUNT
RISES, and by two different amounts for two different reasons. Sixteen of the
seventeen captions were written by hand as reviewed records in all nine
locales, so each gains 16; 'Cellpose' has no record, because a name the gates
keep as it is would be an exact copy. Five locales gain more, because writing
those records exposed existing labels on the same card that said another word,
and each correction is a record only where the old row was wrong: fr +1
(Overlap read 'Rupture'), ko +2 (Overlap 오프화이트, off-white; Bright 밝은 ->
밝음), hi +3 (Overlap ओवरपॉइंट, a non-word; Bright; Object operations read
'purpose of operation'), zh_CN +3 (Overlap 超越, surpass; Bright 光明; objects
物品, goods) and is +5 (Overlap 'Að yfirgefa', to leave; Bright 'ljósi';
Object operations, object and objects, all read as 'Ástæður', reasons). That
is 16 x 9 + 14 = 158 records, 1,377 -> 1,535, and no record left any locale.
The rebuilt catalogs moved those 17 rows and those 14 values and nothing
else, measured table by table against the committed catalogs.*

*Regenerated 2026-09-14, second pass, for instruction 364's `grna`
retirement. The runtime denominator moves 5,664 -> 5,662 and NOT by a flat
two deletions: `grna`'s setting label and its tooltip leave (-2), the
`PATH_LIST_TITLES` chooser caption 'Choose the gRNA CSV' leaves with the
control it named (-1), and the identity 'gRNA' ARRIVES as a UI record (+1).
It arrives because of the removal: while `grna` was a setting whose English
label was 'gRNA', that identity was materialised by the label table and
excluded from the UI sources; retiring the setting hands it to UI instead.
NO REVIEWED RECORD CHANGED in any locale -- the nine reviewed counts are
exactly what they were, because `grna` had no reviewed record in any
language. Its sibling `barcodes` DOES (zh_CN, `setting_tooltips`), which is
why `barcodes` is approved for the same retirement and is not in this pass.
Only zh_CN's percentage moves, 4.71% -> 4.72%, and that is 267 divided by a
denominator two smaller rather than any change in what has been read.*

*Regenerated 2026-09-14. The API denominator moved 10,478 -> 10,531 as well:
53 public symbols arrived and none left, set-differenced against the
baseline rather than subtracted -- 44 from the new curation-queue modules,
7 from the default-graph-type work and 2 from the figure-font work.
The runtime denominator moved 5,361 -> 5,664: 283
captions reached the extractor through fourteen runtime registries that an AST
walk could only see as a variable, and 21 more were the regression-model menu,
which `settings_model` DECLARED for exactly this purpose and which nothing
consumed -- so all 21 read English in all nine locales. Three API counts fall
by one or two (de 424 -> 422, pt 448 -> 446, is 1,025 -> 1,022): protecting the
twenty UI screen names invalidated reviewed records that had TRANSLATED a
screen name, which sent a reader to a screen their interface does not have.
Those records were wrong the day they were written; nothing could see it until
the term was protected. Two locales gain a runtime record (zh_CN 266 -> 267,
hi 126 -> 128) for the three em-dash definition rows no model would take.*

*Regenerated 2026-09-13. The runtime denominator moved 5,244 -> 5,361 because
a one-line `_set_status` wrapper had hidden 117 user-facing status captions
from the extractor across twenty-three screens; they are now declared and
translated in all nine locales. NO REVIEWED RECORD CHANGED -- the reviewed
counts are what they were, and the percentages fall only because the surface
they are measured against grew. That is the honest direction for this number
to move: the captions were always there and always unreviewed, and the report
now counts them.*

*Regenerated 2026-09-11 against the tree after the merge from `main`. Both
denominators moved -- the runtime one DOWN, 5,260 to 5,249, because 364
deleted settings and their captions went with them, and the API one UP,
10,306 to 10,339, with the modules 370, 372, 377, 386, 387 and 388 added.
Several locales lose a record or two for the same reason a setting's caption
left: a reviewed record whose source is no longer in the catalog is not
evidence about anything. The one addition is `press Escape to close`, claimed
in all nine after a rebuild had replaced the key name with a verb in eight of
them.*

## Who reviewed what, and what that claim means

**The maintainer (Einar Olafsson) reads Icelandic, Swedish and German**, and
said so explicitly when asked on 2026-09-02. Only those three locales carry any
human-native review, and only for the strings he was shown.

**On 2026-09-04 he directed that the remaining pass be done and recorded as
reviewed.** That decision is honoured, and its provenance is recorded rather
than blurred: rows added that day were drafted by Claude and accepted by the
maintainer. `docs/i18n/reviewed/runtime/*/2026-09-04-load-family.json` carries
`drafted_by`, `accepted_by` and an `acceptance` note saying in terms that this
is a maintainer's acceptance and **not** a native-speaker review, and that he
did not read fr, hi, ko or zh_CN.

**Hindi (hi): kept and shipped, NOT native-reviewed** -- the maintainer's
decision 2026-09-15, "Keep, marked unreviewed (Recommended)"; to be
reviewed when a native reader is found. Asked with the question tool after
about 200 Hindi UI translations were written that day (201 records in the
sixteen `docs/i18n/reviewed/runtime/hi/2026-09-15-*.json` files), none of
them read by a Hindi speaker. Hindi's row in the table above counts
source-bound evidence, not native review; the language is not hidden.

**The honest public claim remains the one 316 proposed:** "nine languages,
machine-drafted and technically reviewed" is defensible. "Translated into nine
languages" is not, and should not be written in the README, the paper or the
Zenodo record until a reader per locale has passed over them.

## Explicit fallbacks and allowlisted untranslated terms

* `build_i18n_catalogs._IDENTITY_TEXT` — 84 terms that stay English by
  decision (GPU, CSV, gRNA, Amsgrad, Eps and similar). These are excluded from
  the exact-English gate through `_looks_translatable`, which is why a naive
  equality count of "untranslated" rows overstates the work by roughly fifty
  times; measured with the gate's own filter the true remainder was 16, and it
  is now 0.
* `tools/i18n_reviewed_ui.py` — 87 context-pinned UI sources, of which 21 are
  **reviewed identities**: the correct translation is byte-identical to the
  English. `fr: "Source"`, `fr: "Figure"`, `de: "Well"`, `de: "Wells"` and
  `sv/de/pt: "Gate"` are of this kind, and each is an explicit decision rather
  than an untranslated leftover.
* `API_EXACT_TEXT_ALLOWLIST` in `tests/qt/test_external_i18n_catalogs.py` — 17
  API symbols whose rendered text is legitimately identical in some locale.
  Audited entry by entry on 2026-09-04 and reduced from 19: one entry was
  translated in every locale and one named a symbol that no longer exists.

## Defects found and their disposition

Every defect found on 2026-09-04 is listed with what was done about it.

| Defect | Locales | Disposition |
|---|---|---|
| "load" rendered as 충전, to recharge a battery | ko (38 strings) | repaired in the visible label family |
| "load" rendered as "Láttu upp" (let/make) | is | repaired |
| `'Load a column mapping'` rendered "폴더 폴더 폴더 폴더" | ko | repaired |
| "Object" rendered "Syfte" (purpose) | sv | repaired |
| "Model zoo" rendered 动物园模型 (a model of a zoo) | zh_CN | repaired to 模型库 |
| plaque family rendered 板 / 板块 (board, tectonic plate) | zh_CN | repaired to 蚀斑 |
| a microplate well rendered as a water well, the sea, or the adverb | de 5, is 10, ko 2, hi 4 | 21 rows retranslated whole |
| composed setting names left their suffix untranslated in every locale | all nine | fixed in `_composite_translation` |
| `intermedeate_save` still asserted the setting "has no effect" after the English was rewritten | all nine | retranslated |
| `SETTING_LABELS['src']` rendered as a road (ko 도로), a lane (fr Voie), a route (is leiðin) | all nine | retranslated to the Source term |
| "call fields" rendered as TELEPHONE fields | ko, zh_CN (API catalog) | recorded, not yet repaired — see below |

## What is knowingly not done

* **The API catalog's own false friends.** `spacr.regression_panels.apply_primary_call`
  renders "call fields" as 전화 필드 (ko) and 电话字段 (zh_CN) — a telephone call,
  where the sense is a statistical call. It is fluent, passes every gate, and is
  in the API catalog rather than the runtime one, which is a surface no
  systematic reading has covered. Filed against instruction 316.
* **A per-locale fluent reader.** Six of nine locales have never had one, and
  the 2026-09-04 acceptance does not change that.

---

## Table regenerated 2026-09-13

The denominators moved with the Map Barcodes merge and the week's settings
work: runtime 5,265 -> 5,244 sources, API 10,400 -> 10,478 symbols. The test
that guards this table derives both from the builders, so it went red until
this was regenerated -- which is the contract, not a defect.

**SEVERAL REVIEWED COUNTS FELL, and that was checked rather than assumed.**
Spanish 114 -> 99, Portuguese 107 -> 99, Korean 250 -> 245, Chinese 266 -> 262.
A falling count of reviewed records is the signature of STRANDED EVIDENCE -- a
record keyed to a source that has been renamed or removed -- which has happened
in this repository before and cost 32 records.

It is not what happened here:

  * `tools/check_reviewed_runtime_evidence.py` reports every reviewed runtime
    record still matching its source, 1,287 checked, exit 0.
  * The catalogs' reviewed sections were not touched by the 2026-09-13
    rebuild or repair at all: `git diff` on `es`, `pt`, `ko` and `zh_CN`
    contains ZERO lines mentioning `reviewed`.

So the fall happened between 2026-09-04 and now, as 391 retired settings and
renamed others, and the records for settings that no longer exist went with
them. The evidence was not lost; the things it was evidence ABOUT were
withdrawn.

Reviewed totals today: sv 319, de 282, es 284, zh_CN 596, pt 294, hi 383, ko 502, is 451, fr 310 -- 3,421 runtime records across nine locales.

REGENERATED 2026-09-20, and the totals fell rather than rose. 144 runtime
records over nine locales were retired because the English they were
reviewed against no longer exists in spaCR: 15 sources renamed or
rewritten out by items 417, 419, 423 and 435, and 5 whose wording changed
under an unchanged key. On the API side 165 records over 13 rewritten
docstrings were marked `retired` with a reason each, which keeps the
reviewer and the translation as evidence while no longer comparing them
against a docstring that does not contain that sentence. The runtime side
has no such flag -- its loader accepts five fields and nothing else -- so
there the records were removed, which is what items 417 (f6c511cc3) and
418 (c0b2c5227) did in the same situation. Both denominators also moved:
runtime 5,773 to 5,993 and API 10,539 to 10,931, as the work of the last
fortnight added strings and docstrings.
