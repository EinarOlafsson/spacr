# Item spec: in-situ decode of sgRNA spacer prefixes (CROP-seq style): N-cycle prefix whitelist from the library, cycle mask, per-cycle crosstalk compensation, exact then Levenshtein/Hamming ≤ 1 with uniqueness, Perfect/one-error/ambiguous fractions and a per-cell quality class

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs IDENTICALLY (same reads, same calls, same per-cell table). Every seed is fixed. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes (PERISCOPE CP186A genotyping; Feldman experiment C).
- Related: `ITEM_ops_decode_rule_a_s2.md` (decode rule (a), S2, decoy FDR; this item adds the library model and must keep those rules available), `ITEM_cpg_cropped_reader.md` (input), `ITEM_ops_false_call_model.md` / `ITEM_ops_genotype_confidence.md` (consume the outputs).

## Why
Funk 2022 reads a separate 12-nt barcode per guide. PERISCOPE reads the sgRNA spacer itself: the library (`Barcodes.csv`, column `sgRNA`, 20 nt) is the whitelist, and only the first N bases are sequenced and used — CellProfiler `CallBarcodes` uses **10 cycles for A549** and **12 for HeLa** (PCPIP `9_Analysis.cppipe`: `Number of cycles:10` for A549, `12` for HeLa), although 12 cycles were imaged for both. Prefixes of different guides can collide at short N, so the whitelist must be the set of unique prefixes, with collisions reported. Feldman 2019 experiment C read cycles 1–10 with cycle 6 missing (9 cycles; `ITEM_idr0071_stack_reader_decode.md`), so a cycle mask is needed.

spaCR already has: base channels in MiSeq order (`ops_base_channels` = `CY3,A594,CY5,CY7`, i.e. G, T, A, C — the same order PERISCOPE images), a library correction with uniqueness (`ops_sbs.correct_to_library`, Hamming ≤ `max_distance`, ambiguous → None, `N` = unmeasured base matching any letter), and the decode rules of `ITEM_ops_decode_rule_a_s2.md`. It has no spacer-prefix library model, no cycle mask setting, no per-cell quality class comparable to CellProfiler's Perfect / Great, and no Levenshtein option.

## Reference behaviour (what CellProfiler did; what spaCR must expose)
- PCPIP `9_Analysis.cppipe`: foci = `Foci` objects; per cycle max-intensity calls over `CorrCycleNN_{A,C,G,T}` (after the `7_BC_Preprocessing.cppipe` compensation step); `CallBarcodes` against `Barcodes.csv` `sgRNA` with gene `gene_symbol`, score = match score; per cell `Metadata_Foci_Barcode_MatchedTo_Barcode`, `_GeneCode`, `_Score_mean`, `_Score_count`, `Metadata_Foci_Cell_Quality` (Perfect / Great / …; `cell_quality_method = simple`), as in the deposited `A549_single_cell_metadata_CP186A_ALLWELLS.csv.gz`.
- Library design note: decode from 11 bases with error correction from 12 (Levenshtein 2) is the library design; the deposited analysis used N = 10 (A549).
- spaCR's own decode stays the default; this item adds the library model and the reporting.

## What to build
1. **Library model** (`spacr.ops_sbs.spacer_prefix_library`): from a library table (`ops_library` path, `ops_library_sequence_col` = `sgRNA`, `ops_library_gene_col` = `gene_symbol`), take the first `ops_prefix_length` bases of each spacer (after removing masked cycles); collapse identical prefixes; prefixes shared by guides of **different genes** are ambiguous (reported in `library_prefix_collisions.csv`, excluded from calls unless `ops_prefix_collision='gene_level'`, which keeps a prefix shared only by guides of one gene as a gene-level call). Minimum pairwise distance of the whitelist reported.
2. **Cycle mask** (`ops_cycle_mask`, list of 1-based cycles): masked positions become `N` in reads and are removed from the prefix; works with the existing `N` handling of `correct_to_library`.
3. **Compensation:** per-cycle 4 × 4 crosstalk matrix estimated from the spots (median of the dominant-channel clusters, as Funk / Feldman `ops` do), or none (`ops_crosstalk='per_cycle'|'global'|'none'`). P1 crops are already corrected by PCPIP 6–7, so P1 runs with `'none'` first and reports the other arm.
4. **Matching:** exact; then `ops_prefix_max_distance` (1) under `ops_prefix_metric` `'hamming'` (default) or `'levenshtein'` (indels across cycles, as CellProfiler); a read equally close to two whitelist entries is ambiguous (never assigned).
5. **Per-cell call and quality:** reads → cell (existing nucleus/cell assignment); per cell the top barcode, its read count and share; class `perfect` (≥ `ops_quality_min_reads` exact reads of one barcode and no competing barcode), `great` (exact top barcode with a minority competitor, or one-error reads only), `ambiguous`, `none`; definitions written into the run folder so they can be compared with CellProfiler's Perfect / Great.
6. **Reports:** per well and site: fraction of reads exact / one-error / ambiguous / out-of-library; fraction of cells with ≥ 1 perfect read; cells per guide and per gene; prefix collisions; out-of-library rate vs the random-barcode control.

## Settings (alpha)
| Key | Default | Meaning |
|---|---|---|
| `ops_library_mode` | `'barcode'` (today) | `'spacer_prefix'` adds this item |
| `ops_library_sequence_col` | `'sgRNA'` | spacer column |
| `ops_library_gene_col` | `'gene_symbol'` | gene column |
| `ops_prefix_length` | 10 | A549 10, HeLa 12 |
| `ops_sbs_cycles` | 12 | cycles imaged |
| `ops_cycle_mask` | [] | 1-based cycles to ignore |
| `ops_prefix_collision` | `'exclude'` | or `'gene_level'` |
| `ops_prefix_metric` | `'hamming'` | or `'levenshtein'` |
| `ops_prefix_max_distance` | 1 | 0 = exact only |
| `ops_crosstalk` | `'none'` for P1 | `'per_cycle'`, `'global'` |
| `ops_quality_min_reads` | 2 | reads for `perfect` |
| `ops_random_barcode_control` | True | decode against a shuffled whitelist (seed `ops_control_seed` 0) |

## Validation data and acceptance
- **Data:** CP186A Well1 (cropped SBS, cycles 1–12 + Cycle01_DAPI; CellProfiler `BarcodeFoci.csv` per site for 85 sites, `Image.csv`), `Barcodes.csv` and `A549_single_cell_metadata_CP186A_ALLWELLS.csv.gz` (196,153 retained Well1 cells; CP186A Perfect 701,386, Great 321,569), `A549_NGS_Counts.csv` (2022_PERISCOPE repository).
- **Acceptance (comparison, reported with controls; no published per-cell rate exists, so these are not equality targets):**
  - % of Well1 cells with ≥ 1 perfect match, against their 196,153 retained cells (their cells with a call / all segmented cells, from `Image.csv` object counts);
  - out-of-library rate lower than the random-barcode control;
  - per-guide cell counts correlate with their NGS counts (their screen-wide R² 0.84) — report R² for spaCR and for their Well1 calls;
  - prefix collision table for N = 10 reproduces the library's own count (computed from `Barcodes.csv`, deterministic).
- **Determinism:** a rerun from the saved settings gives identical reads, calls and per-cell table; the first accepted run becomes the regression reference.
- **Tests (CPU):** synthetic library with two guides sharing a 10-nt prefix (different genes → excluded; same gene → gene-level call); a read with one substitution → corrected; one indel → corrected only under `'levenshtein'`; masked cycle 6 → `N` and still unique; equidistant read → ambiguous.

## Controls
- **Random-barcode decode** (shuffled whitelist of the same length and composition; seed 0).
- **Out-of-library calls** (decoy prefixes absent from the library, as the decoy FDR).
- **Their CellProfiler calls** on the same cells (through `ITEM_ops_reference_mask_matching.md`) as the reference row.

## Catalogue and locales
New documented callables (`spacer_prefix_library`, the matching function, the quality classifier) ship with arrival contracts and 9-locale records in the same commit; each setting has a tooltip.

## Literature
- Ramezani M et al. (2025) Nat Methods 22:621 (PERISCOPE; CallBarcodes).
- Feldman D et al. (2019) Optical pooled screens in human cells. Cell 179:787–799 (CROP-seq spacer reads; 9-cycle decode).
- Datlinger P et al. (2017) Pooled CRISPR screening with single-cell transcriptome readout. Nat Methods 14:297–301 (CROP-seq vector).
- Funk L et al. (2022) Cell 185:4634 (base calling, crosstalk compensation).

## Effort
- **Effort:** M (library model, mask, metric option, quality classes, reports; base calling and assignment exist).
