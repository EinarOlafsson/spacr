# Localization review scope at the 1.5.0.5 release gate

This is an evidence report, not a certificate that every sentence was read by
a fluent speaker.  The catalogs have complete mechanical source coverage, but
the source-bound semantic-review evidence is defect-driven and much smaller
than the shipped corpus.  Instruction 306 therefore must not be closed on the
basis of these numbers alone.

## Inventory and evidence scope

The live runtime inventory contains 4,978 entries: 1,018 setting labels, 1,013
setting tooltips, 192 category explanations, 2,691 UI strings, and 64 module
summaries.  The live API inventory contains 8,838 symbol documents: 8,719
canonical documents plus 119 rendered aliases.  The nine supported non-English
locales are Swedish
(`sv`), German (`de`), Spanish (`es`), Simplified Chinese (`zh_CN`), Portuguese
(`pt`), Hindi (`hi`), Korean (`ko`), Icelandic (`is`), and French (`fr`).

The table counts unique, current source strings admitted through
`docs/i18n/reviewed/runtime/<locale>/` and unique, current translatable blocks
admitted through `docs/i18n/reviewed/api/<locale>/`.  Percentages and remainders
use the requested release denominators (4,978 runtime entries and 8,838 API
symbol documents).  API evidence is block-keyed, not document-keyed, so its
percentage is an intentionally conservative normalization, not the proportion
of whole API documents read.  Repeated source strings likewise mean that the
runtime percentage is not a unique-string percentage.

| Language | Reviewed runtime sources | Of 4,978 | Arithmetic remainder | Reviewed API blocks | Of 8,838 | Arithmetic remainder |
|---|---:|---:|---:|---:|---:|---:|
| Swedish | 111 | 2.23% | 4,867 | 252 | 2.85% | 8,586 |
| German | 81 | 1.63% | 4,897 | 206 | 2.33% | 8,632 |
| Spanish | 88 | 1.77% | 4,890 | 84 | 0.95% | 8,754 |
| Simplified Chinese | 235 | 4.72% | 4,743 | 236 | 2.67% | 8,602 |
| Portuguese | 92 | 1.85% | 4,886 | 191 | 2.16% | 8,647 |
| Hindi | 100 | 2.01% | 4,878 | 160 | 1.81% | 8,678 |
| Korean | 218 | 4.38% | 4,760 | 186 | 2.10% | 8,652 |
| Icelandic | 106 | 2.13% | 4,872 | 681 | 7.71% | 8,157 |
| French | 89 | 1.79% | 4,889 | 188 | 2.13% | 8,650 |

In addition, `tools/i18n_reviewed_ui.py` pins 84 context-sensitive UI sources
in every locale: 84 x 9 = 756 reviewed source/target pairs.  Those rows overlap
the runtime inventory and are therefore not added to the table.  The review
evidence schemas do not consistently name a fluent-speaker reviewer; some
records and the Portuguese API evidence explicitly describe Codex-assisted
review.  No fluent-speaker census of all 44,802 runtime target entries or all
79,542 API target documents is recorded.

## Sampling method and exact unsampled remainder

Review was purposive, not random: model-rejected hard tails, scientific false
friends, source changes, known UI ambiguity, former tooltip waivers, and defects
found while rendering were selected.  The arithmetic remainders in the table
are exact against the requested denominators, but they are not proof that each
remaining entry is wrong or wholly unreviewed: one reviewed API block can occur
in several documents, and static reviewed vocabularies overlap external review
files.  There was no unnamed statistical sample and no claim of exhaustive
semantic review.

The final settings-default pass records the Regression
`guide_nuisance_columns` tooltip in every locale after model output confused a
microplate well with the adjective “well” in several languages.  Korean also
has reviewed `loss_type` and `normalize` records where protected syntax caused
the model candidates to fail closed.  Each record is bound to the exact current
source and passes the same production syntax, script, and semantic gates as the
generated catalogs.

The final source-bound repair reviewed all four blocks of
`spacr.__main__.main` in each language: two unchanged source blocks plus the
new successful process-exit-code and `SystemExit(2)` parser-error contract
blocks.  It performed no model decoding.

The subsequent Home repair reviewed both blocks of
`spacr.qt.widgets.home.SystemPanel` in every language after its lightweight
probe contract changed.  Those 18 source-bound records also used no model
decoding.

The filters parameter sweep reviewed seven newly documented required-parameter
blocks in Spanish, Simplified Chinese, Hindi, Korean, Icelandic, and French.
Those 42 source-bound records replace model output that retained English or
reversed missing-file behaviour.  Portuguese additionally records the
source-bound repair of an existing `object_type` block that incremental layout
had left in English.  Other regenerated changes were mechanically audited but
are not added to the reviewed count.

The final hard-tail review records the FlowView `NodeItem` card description in
Swedish, German, Spanish, Simplified Chinese, Icelandic, and French; the
even-odd ray-casting description for Spanish `points_in_polygon`; and three
technical blocks from the Simplified Chinese Freedman-Lane guide-permutation
documentation.  It also records the Portuguese and Icelandic icon lookup
contract, three Hindi worker/measurement-panel blocks, two Korean
measurement-panel blocks, four other Icelandic worker/fractal/refit blocks,
and the French same-instant supersampling contract.  All 22 records are bound
to their exact current source block, contextual model input, and SHA-256 hash,
and pass the protected-literal and target-script gates.

The post-docstring hard-tail review binds the 54 residual symbol/block labels
introduced by the required-parameter sweep, representing 51 unique source
blocks across Swedish, German, Simplified Chinese, Portuguese, Hindi,
Icelandic, and French.  These are short parameter descriptions whose generated
targets failed closed; every reviewed replacement is tied to its exact source,
context, and SHA-256 hash and passes the same syntax, semantic, and target-script
gates as the generated catalogs.

The final drag-and-drop and threshold-gate pass compared the same 73 blocks in
25 changed symbols across all nine locales.  The source-bound files record 73
reviewed replacements in seven locales, 57 in Spanish, and 54 in Portuguese:
622 exact evidence records in total, adding 599 current source/target pairs to
the nine reviewed maps.  The review corrected product names, GUI-screen versus
scientific-screen senses, dropped-versus-deleted paths, numeric bounds and
handles, SQLite URI quoting, and the sole Hindi exact-English residual.

## Explicit English identities and fallbacks

The runtime catalogs preserve this complete 84-item reviewed technical-identity
set exactly in English where it occurs:

`%d px`; `--partition=gpu --gres=gpu:1 --time=12:00:00`; `2D / 3D UMAP`;
`3D`; `<a href="api">API</a>`; `API`; `Amsgrad`; `CPU`; `CSV`; `CUDA`; `CV`;
`Cellpose-SAM`; `DNA`; `EAF1_g1, EAF1_g2`; `EC50`; `Eps`; `FOV`; `FlowView`;
`GPU`; `Huber t`; `JSON`; `MIP`; `ML`; `NaN`; `PDF`; `PNG`; `QC`; `RGB`;
`RNA`; `ROI`; `RdBu_r`; `SAM`; `SHAP`; `SQL`; `TIFF`; `Tensorboard`; `UMAP`;
`ViT`; `X`; `XGBoost`; `Y`; `Z`; `btrack`; `cellpose`; `cividis`; `coolwarm`;
`dst`; `fdr_bh`; `gRNA`; `gRNA CSV`; `image_path`; `inferno`; `log10`; `magma`;
`measurements.db`; `metadata_column_map.json`; `numpyro`; `otsu`; `plasma`;
`png`; `png_list`; `png_path`; `pymc`; `seg_qc`; `slurm`; `spaCR`; `ssh`; `t`;
`torch`; `trackastra`; `trackpy`; `tsne`; `turbo`; `ultrack`; `umap`; `viridis`;
`x`; `xD`; `y`; `{report}`; `|Tutorials|`; `µM`; `µm/pixel`; `■ {note}`.

Reviewed locale-specific spellings that are also byte-identical to English are:

- Swedish: `Toxoplasma`, `Regex`, `Budget`, `Gate`.
- German: `Toxoplasma`, `Regex`, `Radius`, `Folds`, `Budget`, `Gate`, `Well`,
  `Wells`.
- Spanish: `Toxoplasma`, `Regex`, `Coef.`.
- Simplified Chinese: `Regex`.
- Portuguese: `Toxoplasma`, `Regex`, `Gate`, `Box gate`, `Coef.`.
- Hindi: `Regex`.
- Korean: `Regex`.
- Icelandic: `Toxoplasma`, `Regex`.
- French: `Toxoplasma`, `Regex`, `Axes…`, `Budget`, `Gate`, `Figure`, `Type`.

The API catalogs preserve exact English only for these 18 reviewed symbol
documents, whose visible contents are code, identifiers, literals, or data
shapes rather than translatable prose:

- `spacr.align.CanvasSpec.shape`
- `spacr.errors.RunLedger.status`
- `spacr.gene_facts.Segment.text`
- `spacr.hits.HitList.flag_counts`
- `spacr.macro.MacroStep.entry`
- `spacr.qt.iconset.themed_pixmap`
- `spacr.qt.screens.report.ReportScreen.output_format`
- `spacr.qt.settings_search.SettingsSearchBar.level`
- `spacr.qt.widgets.dose_response.DoseResponseResult.status`
- `spacr.qt.widgets.formula.Unary`
- `spacr.qt.widgets.plate_layout.PlateDesign.shape`
- `spacr.qt.widgets.setup_card.SetupCard.mode`
- `spacr.run_compare.HitList.by_key`
- `spacr.runctx.RunContext.__str__`
- `spacr.runctx.SkipRecord.__str__`
- `spacr.schema.field_index`
- `spacr.seg_qc.Scorecard.verdict`
- `spacr.updater.PackageChange.describe`

There are no source-current exact-English setting-tooltip prose rows.  Unknown
runtime strings, stale runtime records, malformed/stale API payloads, and API
fetch failures fall back visibly to canonical English and are not counted as
translations.  At this snapshot there are no unresolved catalog rows using
that fallback; the fallback remains a tested safety path.

## Mechanical and rendered evidence

Both catalog audits enforce live-source keys and hashes, placeholder/markup and
protected-literal parity, expected target scripts, copied-English residue, and
known semantic false-friend families.  Required docs CI runs a clean
`sphinx-build -W -E --keep-going`, preserves Sphinx's exit status, and resolves
every emitted tooltip API link against the fresh output.

Runtime tests drive real catalogs through Qt language switching.  Browser tests
exercise rendering, selector persistence, malformed/stale payload fallback, and
request races with two-symbol synthetic catalogs.  The exhaustive browser
ratchet constructs one AutoAPI-shaped document containing every exact English
symbol ID, then loads all nine complete real catalogs and requires exactly
8,838 translated panels whose IDs equal that full symbol union for each locale.
It therefore drives every catalog document through the frontend without the
cost and weaker state isolation of 8,838 separate Chrome launches.  This is
exhaustive frontend coverage, not exhaustive semantic review.  Dedicated
mutations delete a Swedish tooltip translation and inject a broken tooltip API
URL; the runtime audit and link resolver respectively turn red.  Synthetic
source-staleness and partial-Sphinx failures remain covered.

The former twelve `KNOWN_THIN` tooltip waivers are gone.  Each now passes the
ordinary type, length, non-tautology, default-presence, and link checks and is
separately pinned to implementation facts.  Unit ratchets cover all 110
diameter/radius keys plus explicit `_px` and `_um` keys.  Across registered app
defaults, all 674 parseable tooltip/default comparisons are pinned: 622 match
directly and all 52 module-specific variants have explicit expected values and
review reasons.  Of those variants, 21 were already accurate and 31 prompted
shared-tooltip repairs; the four configuration defects found by the review were
also repaired, leaving zero known configuration defects in this inventory.
Forty-seven inactive real-default dependency cases spanning 33 settings must
also state which source setting made them inapplicable.  These are concrete
mechanical advances, not proof that every remaining unit or conditional
sentence outside these ratchets has been semantically reviewed.

API parameter checks reject ghost `:param:` names and contradictory literal
defaults.  Required generated dataclass and NamedTuple fields count an exact
rendered `:ivar:` description; ordinary callables cannot use that rule.
Curated AutoAPI aliases count at their canonical target only after signatures
and rendered prose match exactly, while canonical debt remains counted once.
After those narrow boundary rules, the reverse-direction ratchet records
2,698 required parameters without a structured description across
1,951 public callables.  Its count and digest fail on additions or
substitutions; the debt
is still nonzero.  Its current digest is
``152f999693c3cc9dbfc0b909faa44103a5e2ec87968b41b112db6e24338845d1``.
The 2026-08-30 source-docstring sweep removed 1,015 exact omissions from the
preceding 3,713/2,541 baseline while preserving executable ASTs and zero scoped
ghost fields.

## Disposition

The source/catalog fixed point, warning-as-error documentation gate, complete
real-catalog selector exercise, and explicit mutation ratchets can be green
while the semantic acceptance criteria above remain incomplete.  Item 306 can
move to done only after the 2,698 required-parameter omissions are resolved, the
remaining tooltip unit/default/applicability scope is semantically disposed,
and a named exhaustive or exactly sampled semantic review is completed.
